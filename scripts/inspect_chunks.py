"""Prints how each document is chunked, without calling OpenAI (free, offline).

    python scripts/inspect_chunks.py                       # summary for every document
    python scripts/inspect_chunks.py hr/leave-policy.md    # full text of one file's chunks

Use it after adding documents or changing the chunk settings in config.yaml, to check
that procedures and tables stay whole and that no chunk is a meaningless fragment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunking import chunk_document, count_tokens  # noqa: E402
from rag.config import get_settings  # noqa: E402
from rag.loaders import SUPPORTED_EXTENSIONS, LoaderError, load_file  # noqa: E402


def main() -> int:
    settings = get_settings()
    docs = settings.documents_dir
    only = sys.argv[1] if len(sys.argv) > 1 else None

    files = [p for p in sorted(docs.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    all_sizes = []
    for path in files:
        rel = path.relative_to(docs).as_posix()
        if only and rel != only:
            continue
        try:
            text = load_file(path)
        except LoaderError as exc:
            print(f"{rel}: skipped ({exc})")
            continue
        chunks = chunk_document(
            text, path.stem, settings.chunk_size, settings.chunk_overlap, settings.min_chunk_tokens
        )
        sizes = [count_tokens(c.text) for c in chunks]
        all_sizes += sizes
        print(f"\n{rel}: {len(chunks)} chunks, tokens min {min(sizes)} / max {max(sizes)}")
        for n, (c, size) in enumerate(zip(chunks, sizes)):
            print(f"  #{n:<3} {size:>4} tok  {c.section}")
            if only:
                print("       " + c.text.replace("\n", "\n       ") + "\n")

    if all_sizes and not only:
        print(
            f"\nTotal: {len(all_sizes)} chunks, average {sum(all_sizes) / len(all_sizes):.0f} tokens, "
            f"limit {settings.chunk_size}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
