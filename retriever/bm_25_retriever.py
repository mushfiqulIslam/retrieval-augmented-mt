import logging
import re
import time
from typing import List, Dict

import numpy as np
from rank_bm25 import BM25Okapi

from retriever.base import BaseRetriever

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    def _build_index(self) -> None:
        logger.info(f"Building BM25 index over {len(self.corpus)} documents")
        start = time.perf_counter()

        self._tokenized = [self._tokenize(doc["text"]) for doc in self.corpus]
        self._bm25 = BM25Okapi(
            self._tokenized,
            k1=self.cfg.bm25_k1,
            b=self.cfg.bm25_b,
        )
        elapsed = time.perf_counter() - start
        logger.info(f"BM25 index built in {elapsed:.3f}s. k1={self.cfg.bm25_k1}, b={self.cfg.bm25_b}.")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Rank all documents
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            doc = self.corpus[idx]
            results.append({
                "doc_id": doc.get("id", str(idx)),
                "title":  doc.get("title", ""),
                "text":   doc["text"],
                "score":  float(scores[idx]),
                "rank":   rank,
            })
        return results