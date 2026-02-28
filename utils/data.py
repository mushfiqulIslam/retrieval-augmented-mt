import logging
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from datasets import load_dataset

from utils.config import DataConfig
from utils.sample_corpus import BUILTIN_CORPUS

logger = logging.getLogger(__name__)

@dataclass
class TranslationResult:
    source:       str
    reference:    str
    hypothesis:   str
    context:      Optional[str] = None
    context_tokens: int = 0
    system_name:  str = ""


@dataclass
class MetricScores:
    system_name:          str
    bleu:                 float = 0.0
    comet:                float = 0.0
    comet_metric_name:    str   = "comet"
    hallucination_rate:   float = 0.0
    avg_context_tokens:   float = 0.0
    quality_per_token:    float = 0.0
    n_sentences:          int   = 0
    metadata:             Dict  = field(default_factory=dict)


def load_test_set(cfg: DataConfig, seed: int = 42) -> List[Dict]:
    try:
        logger.info(f"Loading test set from HuggingFace: {cfg.dataset_name} ({cfg.dataset_config})")
        ds = load_dataset(cfg.dataset_name, cfg.dataset_config, trust_remote_code=True)
        split = ds.get("test", ds.get("validation", ds["train"]))

        pairs = []
        for item in split:
            trans = item["translation"]
            pairs.append({"en": trans["en"], "fi": trans["fi"]})

        rng = random.Random(seed)
        rng.shuffle(pairs)
        pairs = pairs[:cfg.test_size]
        logger.info(f"Test set loaded: {len(pairs)} sentence pairs.")
        return pairs

    except Exception as e:
        logger.warning(f"Could not load from HuggingFace ({e}).")
        exit(1)


def load_retrieval_corpus(cfg: DataConfig) -> List[Dict]:
    """
    Load the English retrieval corpus.

    Returns:
        List of {"id": str, "title": str, "text": str} dicts.
    """
    if cfg.corpus_source == "builtin":
        logger.info(f"Using built-in retrieval corpus ({len(BUILTIN_CORPUS)} documents).")
        return BUILTIN_CORPUS

    if cfg.corpus_source == "hf_dataset":
        try:
            from datasets import load_dataset
            logger.info("Loading retrieval corpus from HuggingFace (opus-100 English side).")
            ds = load_dataset("Helsinki-NLP/opus-100", "en-fi", trust_remote_code=True)
            train = ds["train"]
            docs = []
            for i, item in enumerate(train.select(range(cfg.corpus_size))):
                docs.append({
                    "id": f"corp_{i:04d}",
                    "title": f"Document {i}",
                    "text": item["translation"]["en"]
                })
            logger.info(f"Corpus loaded: {len(docs)} documents.")
            return docs
        except Exception as e:
            logger.warning(f"HuggingFace corpus load failed ({e}). Falling back to built-in.")
            return BUILTIN_CORPUS

    return BUILTIN_CORPUS