"""Runs the test questions in eval/golden.jsonl and writes eval/report.md.

    python scripts/evaluate.py              # retrieval + answers
    python scripts/evaluate.py --no-llm     # retrieval metrics only (cheaper, no chat calls)

Measured automatically:
  - Retrieval hit rate@k: at least one expected source section was retrieved.
  - MRR: 1 / rank of the first expected section (1.0 = it was the top result).
  - Declining: unanswerable questions should be declined, answerable ones should not.
Correctness is graded by hand: the report has a "Grade" column for you to fill in.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import PROJECT_ROOT, get_settings  # noqa: E402
from rag.generation import cited_numbers, generate_answer, is_declined  # noqa: E402
from rag.index import index_status, sync_index  # noqa: E402
from rag.retrieval import retrieve  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval"


def matches(chunk, expected: dict) -> bool:
    # Labels use document + section (not chunk ids), so they survive re-chunking.
    # A chunk matches if it is from the expected file and one of its sections starts
    # with the expected section (so "Parental Leave" also matches "Parental Leave > Eligibility").
    # Merged chunks list several sections joined with " | ".
    if chunk.source != expected["source"]:
        return False
    wanted = (expected.get("section") or "").lower()
    return any(s.strip().lower().startswith(wanted) for s in chunk.section.split(" | "))


def first_hit_rank(chunks, expected_sources: list[dict]) -> int | None:
    for rank, chunk in enumerate(chunks, start=1):
        if any(matches(chunk, e) for e in expected_sources):
            return rank
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="skip answer generation")
    args = parser.parse_args()
    settings = get_settings()

    print("Syncing index...")
    sync_index(progress=print)  # make sure the index matches the documents being tested

    questions = [
        json.loads(line)
        for line in (EVAL_DIR / "golden.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    results = []
    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {q['question']}")
        start = time.perf_counter()
        chunks = retrieve(q["question"])
        record = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "answerable": bool(q["expected_sources"]),
            "retrieved": [f"{c.source} | {c.section} | {c.score:.2f}" for c in chunks],
            "top_score": chunks[0].score if chunks else 0.0,
        }
        if record["answerable"]:
            rank = first_hit_rank(chunks, q["expected_sources"])
            record["hit"] = rank is not None
            record["reciprocal_rank"] = 1.0 / rank if rank else 0.0
        if not args.no_llm:
            answer = generate_answer(q["question"], chunks)
            record["answer"] = answer
            record["declined"] = is_declined(answer)
            record["cited"] = sorted(cited_numbers(answer))
        record["seconds"] = round(time.perf_counter() - start, 2)
        results.append(record)

    # Save raw results so they can be compared between runs.
    results_dir = EVAL_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (results_dir / f"run-{stamp}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results), encoding="utf-8"
    )

    report = build_report(results, settings, args.no_llm)
    (EVAL_DIR / "report.md").write_text(report, encoding="utf-8")
    print(f"\nWrote {EVAL_DIR / 'report.md'}")
    return 0


def _rate(values: list[bool]) -> str:
    return f"{sum(values) / len(values):.2f} ({sum(values)}/{len(values)})" if values else "n/a"


def build_report(results: list[dict], settings, no_llm: bool) -> str:
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]
    stats = index_status()
    mrr = sum(r["reciprocal_rank"] for r in answerable) / len(answerable) if answerable else 0.0

    lines = [
        "# Evaluation Report",
        "",
        f"**Run:** {datetime.now():%Y-%m-%d %H:%M}  ",
        f"**Index:** {stats['documents']} documents, {stats['chunks']} chunks  ",
        f"**Config:** chat model `{settings.chat_model}`, embeddings `{settings.embedding_model}`, "
        f"chunk {settings.chunk_size}/{settings.chunk_overlap} tokens, top k {settings.top_k}, dense search only",
        "",
        "## Metrics",
        "",
        "| Metric | Result | Target |",
        "|---|---|---|",
        f"| Retrieval hit rate@{settings.top_k} | {_rate([r['hit'] for r in answerable])} | ≥ 0.90 |",
        f"| MRR | {mrr:.2f} | ≥ 0.70 |",
    ]
    if not no_llm:
        lines += [
            f"| Unanswerable questions declined | {_rate([r['declined'] for r in unanswerable])} | ≥ 0.90 |",
            f"| Answerable questions NOT declined | {_rate([not r['declined'] for r in answerable])} | ≥ 0.90 |",
            "| Correctness (hand-graded) | fill in from the table below | ≥ 0.85 |",
        ]
    seconds = sorted(r["seconds"] for r in results)
    lines += [
        f"| Median time per question | {seconds[len(seconds) // 2]:.2f}s | report only |",
        "",
        "## Results by question",
        "",
        "Grade each answer by hand as `correct`, `partial` or `incorrect` in the Grade column.",
        "",
    ]
    for r in results:
        status = []
        if r["answerable"]:
            status.append("hit" if r["hit"] else "**MISS**")
        if "declined" in r:
            status.append("declined" if r["declined"] else "answered")
        lines += [
            f"### {r['id']} ({r['category']}): {r['question']}",
            "",
            f"- **Status:** {', '.join(status)} · top similarity {r['top_score']:.2f}",
            f"- **Expected:** {r['expected_answer']}",
        ]
        if "answer" in r:
            lines.append(f"- **Answer:** {r['answer'].strip()}".replace("\n", "\n  "))
        lines += [
            "- **Retrieved:** " + "; ".join(r["retrieved"][:3]),
            "- **Grade:** ",
            "",
        ]
    lines += [
        "## Limitations",
        "",
        "- One small test set with no separate dev/test split, so the settings were not tuned "
        "against held-out questions.",
        "- Dense search only. BM25 hybrid search, a reranker and a relevance gate are deferred (DESIGN.md section 14).",
        "- Sources show what was retrieved and the numbers the model cited, not verified per-claim citations.",
        "- Correctness is hand-graded by one person.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
