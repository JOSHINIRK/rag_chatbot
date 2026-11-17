"""Checks the chunking rules that matter for retrieval accuracy. Run: python -m pytest"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunking import chunk_document, count_tokens  # noqa: E402

SIZE, OVERLAP, MIN = 120, 30, 60


def chunks_of(text, size=SIZE, overlap=OVERLAP, min_tokens=MIN):
    return chunk_document(text, "fallback", size, overlap, min_tokens)


def test_every_chunk_starts_with_its_heading_path():
    chunks = chunks_of("# Policy\n\n## Leave\n\n### Sick\n\nCall your manager before 10:00.")
    assert chunks[0].text.startswith("Policy > Leave > Sick\n\n")
    assert chunks[0].title == "Policy"
    assert chunks[0].section == "Leave > Sick"


def test_document_without_headings_uses_file_name_as_title():
    chunks = chunks_of("Just some text without headings.")
    assert chunks[0].title == "fallback"


def test_numbered_steps_with_blank_lines_stay_in_one_chunk_with_their_lead_in():
    text = "# Guide\n\n## Reset\n\nTo reset your password:\n\n1. Open the site.\n\n2. Enter your email.\n\n3. Approve in the app."
    chunks = chunks_of(text)
    assert len(chunks) == 1
    body = chunks[0].text
    assert "To reset your password:" in body and "1. Open the site." in body and "3. Approve in the app." in body


def test_long_prose_is_split_at_sentence_boundaries_with_overlap_and_within_size():
    sentences = [f"Rule number {i} says employees must follow procedure {i} carefully." for i in range(40)]
    chunks = chunks_of("# Policy\n\n## Rules\n\n" + " ".join(sentences))
    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c.text) <= SIZE
        body = c.text.split("\n\n", 1)[1]
        assert body.endswith("carefully.")  # never cut mid-sentence
    # Overlap: the next chunk repeats the last sentence of the previous one.
    last_sentence = chunks[0].text.rsplit("Rule number", 1)[1]
    assert ("Rule number" + last_sentence) in chunks[1].text


def test_big_table_is_split_by_rows_and_repeats_the_header():
    rows = "\n".join(f"| item {i} | value {i} for the item |" for i in range(60))
    text = f"# IT\n\n## Prices\n\n| Item | Value |\n|---|---|\n{rows}"
    chunks = chunks_of(text)
    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c.text) <= SIZE
        assert "| Item | Value |\n|---|---|" in c.text
    joined = "".join(c.text for c in chunks)
    assert all(f"| item {i} |" in joined for i in range(60))  # no row lost


def test_hash_lines_inside_code_blocks_are_not_headings():
    text = "# Setup\n\n## Script\n\n```\n# not a heading\necho hi\n```"
    chunks = chunks_of(text)
    assert len(chunks) == 1
    assert chunks[0].section == "Script"
    assert "# not a heading" in chunks[0].text


def test_tiny_sections_on_the_same_topic_are_merged():
    text = (
        "# Leave\n\n## Parental Leave\n\n### Eligibility\n\nAfter 6 months.\n\n"
        "### Duration\n\n16 weeks at full pay.\n\n## Bereavement\n\nUp to 5 days."
    )
    chunks = chunks_of(text)
    parental = [c for c in chunks if "Parental Leave > Eligibility" in c.section]
    assert len(parental) == 1
    assert "16 weeks" in parental[0].text  # Duration merged with Eligibility
    assert "Up to 5 days" not in parental[0].text  # a different topic is never merged in
    assert parental[0].text.startswith("Leave > Parental Leave\n\n")


def test_preamble_joins_only_the_first_topic():
    text = "# Policy\n\nIntro line.\n\n## A\n\nText A.\n\n## B\n\nText B."
    chunks = chunks_of(text)
    assert chunks[0].section == "Introduction | A"
    assert "Text B." not in chunks[0].text


def test_large_sections_are_not_merged():
    para = " ".join(f"Sentence {i} about the topic." for i in range(20))
    text = f"# Doc\n\n## A\n\n### One\n\n{para}\n\n### Two\n\n{para}"
    sections = [c.section for c in chunks_of(text)]
    assert not any("|" in s for s in sections)


def test_pathological_long_word_is_still_cut_to_size():
    chunks = chunks_of("# Doc\n\n## Link\n\n" + "x" * 3000)
    assert all(count_tokens(c.text) <= SIZE for c in chunks)


def test_sample_documents_chunk_within_size():
    root = Path(__file__).resolve().parent.parent / "documents"
    for path in root.rglob("*.md"):
        for c in chunk_document(path.read_text(encoding="utf-8"), path.stem, 350, 50, 100):
            assert 0 < count_tokens(c.text) <= 350
