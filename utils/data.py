import logging
import random
from typing import List, Dict

from datasets import load_dataset

from utils.config import DataConfig

logger = logging.getLogger(__name__)

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