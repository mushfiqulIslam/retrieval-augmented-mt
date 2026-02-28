from typing import Dict, Tuple, List

from retriever.base import BaseRetriever


class CachedRetriever:
    """
    Since the same retrieved documents must be used for all context selection
    strategies, retrieved once per (query, top_k) pair and cache the result.
    This guarantees identical retrieval across Systems B and C.
    """
    def __init__(self, retriever: BaseRetriever):
        self._retriever = retriever
        self._cache: Dict[Tuple[str, int], List[Dict]] = {}

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        """Return cached result if available; retrieve and cache otherwise."""
        key = (query, top_k)
        if key not in self._cache:
            self._cache[key] = self._retriever.retrieve(query, top_k)
        return self._cache[key]

    def clear_cache(self):
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)