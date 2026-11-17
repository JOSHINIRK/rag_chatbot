"""Dense (vector) search over the ChromaDB index."""

from dataclasses import dataclass

from rag.config import get_settings
from rag.embeddings import embed_query
from rag.index import get_collection


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    title: str
    section: str
    domain: str
    score: float  # cosine similarity: 1.0 = identical meaning, around 0 = unrelated


def retrieve(question: str, top_k: int | None = None, domain: str | None = None) -> list[Chunk]:
    settings = get_settings()
    top_k = top_k or settings.top_k
    collection = get_collection(settings)
    if collection.count() == 0:
        return []

    # Embed the question with the SAME model used for the chunks, so both live in
    # the same vector space and can be compared.
    query_vector = embed_query(question)

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        # Optional metadata filter, e.g. only HR documents.
        where={"domain": domain} if domain else None,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    # Chroma returns one list per query; we sent one query, hence the [0].
    for chunk_id, text, meta, distance in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=text,
                source=meta["source"],
                title=meta["title"],
                section=meta["section"],
                domain=meta["domain"],
                score=1.0 - distance,  # cosine distance -> cosine similarity
            )
        )
    return chunks  # already ordered from most to least similar
