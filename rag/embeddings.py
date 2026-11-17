"""OpenAI client and text embedding helpers."""

from functools import lru_cache

from openai import OpenAI

from rag.config import get_api_key, get_settings


class MissingAPIKeyError(RuntimeError):
    pass


@lru_cache  # one client per process; it keeps an HTTP connection pool, so reuse it
def get_client() -> OpenAI:
    key = get_api_key()
    if key is None:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and paste your key "
            "(see README.md, 'Get an OpenAI API key')."
        )
    # max_retries: the SDK automatically retries rate-limit (429) and server (5xx)
    # errors this many times, with backoff, before raising an exception.
    return OpenAI(api_key=key, max_retries=2, timeout=60)


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Returns one embedding vector per input text, in the same order."""
    settings = get_settings()
    client = get_client()
    vectors: list[list[float]] = []
    # Send texts in batches: far fewer HTTP calls than one request per chunk.
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=settings.embedding_model, input=batch)
        # Sort by index so vectors line up with their texts, whatever order they arrive in.
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
