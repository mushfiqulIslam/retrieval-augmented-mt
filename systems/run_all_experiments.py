import logging
import os

from utils.config import ExperimentConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_all_experiments(cfg: ExperimentConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.output_dir, "experiment_config.json"))
    logger.info(f"Config saved to {cfg.output_dir}/experiment_config.json")