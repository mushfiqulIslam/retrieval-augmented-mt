import logging
import os
import random

import numpy as np
import torch

from utils.config import ExperimentConfig
from utils.data import load_test_set

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"All random seeds set to {seed}.")


def run_all_experiments(cfg: ExperimentConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_all_seeds(cfg.seed)
    cfg.save(os.path.join(cfg.output_dir, "experiment_config.json"))
    logger.info(f"Config saved to {cfg.output_dir}/experiment_config.json")

    logger.info("\n═════════════════ LOADING DATA ═════════════════")
    test_pairs = load_test_set(cfg.data, seed=cfg.seed)