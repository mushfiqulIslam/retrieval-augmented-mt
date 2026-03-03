import logging
import time
from typing import List, Dict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from retriever.base import BaseRetriever

logger = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    def _build_index(self) -> None:
        logger.info(f"Loading dense model: {self.cfg.dense_model}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(self.cfg.dense_model, device=device)

        logger.info(f"Encoding {len(self.corpus)} documents...")
        start = time.perf_counter()
        texts = [doc["text"] for doc in self.corpus]
        self._doc_embeddings = self._model.encode(
            texts,
            batch_size=self.cfg.dense_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,   # L2-norm → dot product = cosine sim
            convert_to_numpy=True,
        )
        elapsed = time.perf_counter() - start
        logger.info(f"Dense index built in {elapsed:.3f}s. Shape: {self._doc_embeddings.shape}.")

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        query_emb = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        scores = self._doc_embeddings @ query_emb   # (N,) cosine similarities
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