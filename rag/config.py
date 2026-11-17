"""Loads config.yaml and the OpenAI API key into one Settings object."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Project root = the folder that contains config.yaml (one level above rag/).
# Resolving paths from here means the app works no matter which folder you start it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)  # frozen = settings can't be changed by accident at runtime
class Settings:
    documents_dir: Path
    index_dir: Path
    collection_name: str
    embedding_model: str
    chat_model: str
    temperature: float | None
    max_tokens: int
    chunk_size: int
    chunk_overlap: int
    min_chunk_tokens: int
    top_k: int


@lru_cache  # read the file once, then reuse the same Settings object everywhere
def get_settings() -> Settings:
    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)  # safe_load never executes code hidden in YAML

    return Settings(
        # Relative paths in config.yaml are turned into absolute paths under the project root.
        documents_dir=PROJECT_ROOT / raw["documents_dir"],
        index_dir=PROJECT_ROOT / raw["index_dir"],
        collection_name=raw["collection_name"],
        embedding_model=raw["embedding_model"],
        chat_model=raw["chat_model"],
        temperature=raw.get("temperature"),
        max_tokens=int(raw["max_tokens"]),
        chunk_size=int(raw["chunk_size"]),
        chunk_overlap=int(raw["chunk_overlap"]),
        min_chunk_tokens=int(raw.get("min_chunk_tokens", 0)),
        top_k=int(raw["top_k"]),
    )


def get_api_key() -> str | None:
    # Copies the values in .env into environment variables. Variables that are
    # already set in the shell win, so the key can come from either place.
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY", "").strip()
    # Treat the placeholder from .env.example as "no key" so the user gets a clear message.
    if not key or key.endswith("your-key-here"):
        return None
    return key
