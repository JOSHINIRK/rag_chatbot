"""Builds and syncs the ChromaDB index with the documents folder.

Re-indexing is idempotent: each file's SHA-256 hash is stored in index/manifest.json,
so unchanged files are skipped, and all chunks of a changed or deleted file are
removed before new ones are added (stale policy text can't linger).
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import chromadb

from rag.chunking import CHUNKER_VERSION, chunk_document
from rag.config import Settings, get_settings
from rag.embeddings import embed_texts
from rag.loaders import SUPPORTED_EXTENSIONS, LoaderError, load_file


class EmptyDocumentsError(RuntimeError):
    pass


class SettingsChangedError(RuntimeError):
    pass


@dataclass
class SyncSummary:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (file, reason)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)

    def describe(self) -> str:
        text = (
            f"{len(self.added)} added, {len(self.updated)} updated, {len(self.removed)} removed, "
            f"{len(self.unchanged)} unchanged, {len(self.skipped)} skipped"
        )
        for name, reason in self.skipped:
            text += f"\n  skipped {name}: {reason}"
        return text


# ---------------------------------------------------------------- storage helpers


def _manifest_path(settings: Settings) -> Path:
    return settings.index_dir / "manifest.json"


def _index_settings(settings: Settings) -> dict:
    # The settings that change the vectors or chunks. If any differ from the ones the
    # index was built with, old and new chunks would be incompatible -> rebuild needed.
    return {
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "min_chunk_tokens": settings.min_chunk_tokens,
        "chunker_version": CHUNKER_VERSION,  # changes when the chunking code itself changes
    }


def load_manifest(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    path = _manifest_path(settings)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"settings": _index_settings(settings), "files": {}, "last_indexed": None}


def _save_manifest(settings: Settings, manifest: dict) -> None:
    path = _manifest_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and then swap it in, so a crash mid-write can't leave
    # a half-written (corrupt) manifest behind.
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _client(settings: Settings):
    # PersistentClient stores the database on disk under index/chroma, so the index
    # survives restarts and only changed files need re-embedding.
    return chromadb.PersistentClient(path=str(settings.index_dir / "chroma"))


def get_collection(settings: Settings | None = None):
    settings = settings or get_settings()
    return _client(settings).get_or_create_collection(
        name=settings.collection_name,
        # Cosine distance is the right similarity measure for OpenAI embeddings.
        metadata={"hnsw:space": "cosine"},
        # We compute embeddings ourselves with OpenAI, so Chroma must not use its own model.
        embedding_function=None,
    )


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _domain_for(rel_path: str) -> str:
    # The top-level subfolder sets the domain: "hr/leave.md" -> "hr".
    # Files placed directly in documents/ get "general".
    parts = rel_path.split("/")
    return parts[0].lower() if len(parts) > 1 else "general"


# ---------------------------------------------------------------- main sync


def sync_index(
    rebuild: bool = False,
    progress: Callable[[str], None] | None = None,
) -> SyncSummary:
    """Brings the index in line with the documents folder and returns what changed."""
    settings = get_settings()
    report = progress or (lambda msg: None)  # no-op if the caller doesn't want progress
    summary = SyncSummary()
    docs_dir = settings.documents_dir

    # 1. Find all files (recursively) and separate supported from unsupported ones.
    files: dict[str, Path] = {}
    if docs_dir.exists():
        for path in sorted(docs_dir.rglob("*")):
            if not path.is_file() or path.name.startswith("."):  # ignores .gitkeep etc.
                continue
            rel = path.relative_to(docs_dir).as_posix()  # forward slashes on every OS
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files[rel] = path
            else:
                summary.skipped.append((rel, f"unsupported file type {path.suffix or '(none)'}"))

    if not files:
        raise EmptyDocumentsError(
            f"No documents found. Put .md, .txt, .pdf, .docx or .html files in "
            f"{docs_dir / 'hr'} or {docs_dir / 'it'}."
        )

    # 2. Rebuild = wipe the index; otherwise refuse to mix chunks built with different settings.
    if rebuild and settings.index_dir.exists():
        # Drop the collection through Chroma rather than deleting its files: on Windows,
        # files that Chroma still has open can't be deleted.
        try:
            _client(settings).delete_collection(settings.collection_name)
        except Exception:
            pass  # the collection didn't exist yet
        _manifest_path(settings).unlink(missing_ok=True)
    manifest = load_manifest(settings)
    if manifest["settings"] != _index_settings(settings):
        raise SettingsChangedError(
            "The embedding model or chunking settings changed since the index was built. "
            "Run: python scripts/ingest.py --rebuild"
        )
    collection = get_collection(settings)

    # 3. Files that were in the index but are gone from the folder: delete their chunks.
    for rel in list(manifest["files"]):
        if rel not in files:
            collection.delete(where={"source": rel})
            del manifest["files"][rel]
            summary.removed.append(rel)
            _save_manifest(settings, manifest)

    # 4. New or changed files: load, chunk, embed, store.
    for i, (rel, path) in enumerate(files.items(), start=1):
        digest = _file_hash(path)
        previous = manifest["files"].get(rel)
        if previous and previous["hash"] == digest:
            summary.unchanged.append(rel)  # same content as last time -> nothing to do
            continue

        report(f"Indexing {rel} ({i}/{len(files)})")
        try:
            text = load_file(path)
        except LoaderError as exc:
            summary.skipped.append((rel, str(exc)))
            continue

        chunks = chunk_document(
            text, path.stem, settings.chunk_size, settings.chunk_overlap, settings.min_chunk_tokens
        )
        if not chunks:
            summary.skipped.append((rel, "no text after chunking"))
            continue

        vectors = embed_texts([c.text for c in chunks])  # one OpenAI call per 100 chunks

        # Remove the old version's chunks first, so no outdated text stays searchable.
        collection.delete(where={"source": rel})
        collection.add(
            ids=[f"{rel}#{n}" for n in range(len(chunks))],  # stable id: file + position
            documents=[c.text for c in chunks],
            embeddings=vectors,
            metadatas=[
                {
                    "source": rel,
                    "title": c.title,
                    "section": c.section,
                    "domain": _domain_for(rel),
                    "content_hash": digest,
                }
                for c in chunks
            ],
        )

        manifest["files"][rel] = {"hash": digest, "chunks": len(chunks)}
        (summary.updated if previous else summary.added).append(rel)
        _save_manifest(settings, manifest)  # save after every file so progress isn't lost on error

    if summary.changed or manifest["last_indexed"] is None:
        manifest["last_indexed"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _save_manifest(settings, manifest)
    return summary


def index_status() -> dict:
    """Numbers shown in the app sidebar."""
    manifest = load_manifest()
    return {
        "documents": len(manifest["files"]),
        "chunks": sum(f["chunks"] for f in manifest["files"].values()),
        "last_indexed": manifest["last_indexed"],
    }
