import logging
from abc import ABC, abstractmethod
from typing import List

import torch
from sentence_transformers import SentenceTransformer

from utils.config import ContextSelectorConfig

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


class BaseScorer(ABC):

    @abstractmethod
    def score(self, source, sentences) -> List[float]:
        pass


class EmbeddingScorer(BaseScorer):
    """
    Relevance scoring via cosine similarity of sentence embeddings.
    Method:
      1. Encode source and all candidate sentences with a bi-encoder.
      2. Compute cosine similarity between source embedding and each candidate.
      3. Return similarities as relevance scores.

    Deterministic: same model, same input = same output.
    """

    def __init__(self, model_name= "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None   # lazy load

    def _ensure_loaded(self):
        if self._model is None:
            try:
                logger.info(f"Loading embedding scorer model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name, device=device)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for embedding scoring. "
                    "Install with: pip install sentence-transformers"
                )

    def score(self, source, sentences) -> List[float]:
        if not sentences:
            return []
        self._ensure_loaded()

        all_texts = [source] + sentences
        embeddings = self._model.encode(
            all_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        source_emb      = embeddings[0]      # shape: (dim,)
        candidate_embs  = embeddings[1:]     # shape: (N, dim)

        # Cosine similarity (vectors are already L2-normalized → dot product = cosine)
        scores = (candidate_embs @ source_emb).tolist()
        return scores


class LexicalScorer(BaseScorer):
    """
    Lexical relevance scoring via word-overlap (Jaccard similarity).
    Used as a baseline/control in ablation studies.
    score(src, cand) = |tokens(src) ∩ tokens(cand)| / |tokens(src) ∪ tokens(cand)|
    """

    @staticmethod
    def _tokenize_set(text):
        import re
        tokens = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()

        # Remove stopwords
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "in", "on", "at",
            "to", "of", "and", "or", "but", "it", "i", "he", "she", "we",
            "they", "that", "this", "with", "for", "from", "by"
        }
        return {t for t in tokens if t not in stopwords and len(t) > 2}

    def score(self, source, sentences) -> List[float]:
        if not sentences:
            return []
        src_tokens = self._tokenize_set(source)
        scores = []
        for sent in sentences:
            cand_tokens = self._tokenize_set(sent)
            if not src_tokens and not cand_tokens:
                scores.append(0.0)
            elif not src_tokens or not cand_tokens:
                scores.append(0.0)
            else:
                intersection = len(src_tokens & cand_tokens)
                union        = len(src_tokens | cand_tokens)
                scores.append(intersection / union)
        return scores


class CrossEncoderScorer(BaseScorer):
    """
    Cross-encoder relevance scoring (highest quality, slower).

    Uses a cross-encoder model that jointly encodes (source, candidate) pairs
    and produces a relevance score. More accurate than bi-encoder cosine sim.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading cross-encoder model: {self.model_name}")
                self._model = CrossEncoder(self.model_name, device=device)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for cross-encoder scoring."
                )

    def score(self, source, sentences) -> List[float]:
        if not sentences:
            return []
        self._ensure_loaded()
        pairs  = [(source, s) for s in sentences]
        scores = self._model.predict(pairs)
        return scores.tolist()


def build_scorer(cfg: ContextSelectorConfig) -> BaseScorer:
    if cfg.scoring_method == "embedding":
        return EmbeddingScorer(model_name=cfg.embedding_model)
    elif cfg.scoring_method == "cross_encoder":
        return CrossEncoderScorer()
    elif cfg.scoring_method == "lexical":
        return LexicalScorer()
    else:
        raise ValueError(f"Unknown scoring method: {cfg.scoring_method!r}")