"""The full RAG flow for one question: retrieve -> generate.

Each question is answered independently (no conversation memory), as agreed in
DESIGN.md section 2.
"""

from collections.abc import Iterator
from dataclasses import dataclass

from rag.generation import generate_answer, stream_answer
from rag.retrieval import Chunk, retrieve


@dataclass
class StreamingAnswer:
    chunks: list[Chunk]       # the passages the answer is based on (shown as sources)
    stream: Iterator[str]     # the answer text, arriving piece by piece


def answer_stream(question: str, domain: str | None = None) -> StreamingAnswer:
    # Step 1 (Retrieval): find the passages most similar in meaning to the question.
    chunks = retrieve(question, domain=domain)
    # Step 2 (Augmented Generation): give those passages to the LLM to answer from.
    return StreamingAnswer(chunks=chunks, stream=stream_answer(question, chunks))


def answer(question: str, domain: str | None = None) -> tuple[str, list[Chunk]]:
    chunks = retrieve(question, domain=domain)
    return generate_answer(question, chunks), chunks
