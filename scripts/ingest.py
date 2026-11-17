"""Indexes the documents folder from the command line.

    python scripts/ingest.py            # add new/changed files, remove deleted ones
    python scripts/ingest.py --rebuild  # wipe the index and embed everything again

The Streamlit app runs the same sync automatically when it opens, so this script
is only needed for a first build, a rebuild, or checking ingest output.
"""

import argparse
import sys
from pathlib import Path

# Lets "python scripts/ingest.py" import the rag package from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openai  # noqa: E402  (imports must come after the sys.path change above)

from rag.embeddings import MissingAPIKeyError  # noqa: E402
from rag.generation import friendly_error  # noqa: E402
from rag.index import EmptyDocumentsError, SettingsChangedError, index_status, sync_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync the ChromaDB index with documents/.")
    parser.add_argument("--rebuild", action="store_true", help="delete the index and rebuild it from scratch")
    args = parser.parse_args()

    try:
        summary = sync_index(rebuild=args.rebuild, progress=print)
    except (EmptyDocumentsError, SettingsChangedError) as exc:
        print(f"Stopped: {exc}")
        return 1
    except (MissingAPIKeyError, openai.OpenAIError) as exc:
        print(f"Stopped: {friendly_error(exc)}")
        return 1

    stats = index_status()
    print(f"\nDone: {summary.describe()}")
    print(f"Index now holds {stats['documents']} documents in {stats['chunks']} chunks.")
    return 0  # exit code 0 = success, so the script can be used in other scripts


if __name__ == "__main__":
    sys.exit(main())
