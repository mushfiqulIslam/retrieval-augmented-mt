from typing import Dict, Tuple, List

from retriever.base import BaseRetriever


class CachedRetriever:

    def __init__(self, retriever: BaseRetriever):
        self._retriever = retriever
        self._cache: Dict[Tuple[str, int], List[Dict]] = {}