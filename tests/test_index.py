"""Checks index syncing and retrieval offline: OpenAI embeddings are replaced by a fake."""

import dataclasses
import hashlib
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag.config  # noqa: E402
import rag.index as index  # noqa: E402
import rag.retrieval as retrieval  # noqa: E402


def fake_embed(texts, batch_size=100):
    # Bag-of-words hashing: texts sharing words get similar vectors. Good enough to test plumbing.
    vectors = []
    for text in texts:
        v = [0.0] * 256
        for word in re.findall(r"[a-z]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1
        vectors.append(v)
    return vectors


@pytest.fixture
def docs(tmp_path, monkeypatch):
    settings = dataclasses.replace(
        rag.config.get_settings(), documents_dir=tmp_path / "documents", index_dir=tmp_path / "index"
    )
    for module in (index, retrieval):
        monkeypatch.setattr(module, "get_settings", lambda: settings)
    monkeypatch.setattr(index, "embed_texts", fake_embed)
    monkeypatch.setattr(retrieval, "embed_query", lambda t: fake_embed([t])[0])
    (settings.documents_dir / "hr").mkdir(parents=True)
    (settings.documents_dir / "it").mkdir()
    return settings.documents_dir


def test_sync_adds_skips_updates_and_removes(docs):
    (docs / "hr" / "leave.md").write_text("# Leave\n\n## Sick\n\nA medical certificate is needed after 3 days.")
    (docs / "it" / "vpn.md").write_text("# VPN\n\n## Setup\n\nConnect to vpn.example.com.")
    (docs / "notes.xyz").write_text("ignored")

    first = index.sync_index()
    assert sorted(first.added) == ["hr/leave.md", "it/vpn.md"]
    assert first.skipped[0][0] == "notes.xyz"

    second = index.sync_index()
    assert sorted(second.unchanged) == ["hr/leave.md", "it/vpn.md"] and not second.changed

    (docs / "hr" / "leave.md").write_text("# Leave\n\n## Sick\n\nA medical certificate is needed after 5 days.")
    (docs / "it" / "vpn.md").unlink()
    third = index.sync_index()
    assert third.updated == ["hr/leave.md"] and third.removed == ["it/vpn.md"]

    top = retrieval.retrieve("medical certificate")
    assert all(c.source == "hr/leave.md" for c in top)
    assert "5 days" in top[0].text and "3 days" not in top[0].text  # old version is gone
    assert top[0].domain == "hr"


def test_empty_folder_raises(docs):
    with pytest.raises(index.EmptyDocumentsError):
        index.sync_index()


def test_changed_chunk_settings_require_rebuild(docs, monkeypatch):
    (docs / "hr" / "a.md").write_text("# A\n\nText.")
    index.sync_index()
    changed = dataclasses.replace(index.get_settings(), chunk_size=200)
    monkeypatch.setattr(index, "get_settings", lambda: changed)
    with pytest.raises(index.SettingsChangedError):
        index.sync_index()
    assert index.sync_index(rebuild=True).added == ["hr/a.md"]
