"""Response caching: exact (hash-based) and semantic (embedding-based)."""

from caelum.cache.exact import ExactCache, compute_cache_key, should_cache
from caelum.cache.semantic import (
    AsyncEmbedder,
    Embedder,
    EmbeddingResult,
    OpenAIEmbedder,
    SemanticCache,
)
from caelum.cache.stores import CacheStore, SQLiteStore, default_cache_path
from caelum.cache.vectors import SQLiteVectorStore

__all__ = [
    "AsyncEmbedder",
    "CacheStore",
    "Embedder",
    "EmbeddingResult",
    "ExactCache",
    "OpenAIEmbedder",
    "SQLiteStore",
    "SQLiteVectorStore",
    "SemanticCache",
    "compute_cache_key",
    "default_cache_path",
    "should_cache",
]
