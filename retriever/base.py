import logging
from abc import ABC, abstractmethod
from typing import List, Dict

from utils.config import RetrieverConfig

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    def __init__(self, corpus: List[Dict], cfg: RetrieverConfig):
        self.corpus = corpus
        self.cfg    = cfg
        self._build_index()

    @abstractmethod
    def _build_index(self) -> None:
        """Build the retrieval index from the corpus."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        pass