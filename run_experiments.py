import argparse
import logging

from systems.run_all_experiments import run_all_experiments
from utils.config import ExperimentConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def apply_profile(cfg: ExperimentConfig, profile: str) -> None:
    if profile == "default":
        return
    if profile != "presentation":
        raise ValueError(f"Unknown profile: {profile!r}")

    cfg.seed = 42
    cfg.data.test_source = "hf_dataset"
    cfg.data.corpus_source = "builtin"
    cfg.retriever.methods = ["bm25"]
    cfg.retriever.method = "bm25"
    cfg.retriever.top_k_values = [3, 5]
    cfg.context_selector.top_n_values = [1, 3, 5]
    cfg.translator.num_beams = 4
    cfg.run_system_a = True
    cfg.run_system_b = True
    cfg.run_system_c = True
    cfg.run_ablations = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG-MT Experiment Runner:EN→FI with context selection."
    )
    parser.add_argument("--config",    type=str, default=None,
                        help="Path to JSON config file.")
    parser.add_argument("--test_size", type=int, default=None,
                        help="Override test set size.")
    parser.add_argument("--seed",      type=int, default=None,
                        help="Override random seed.")
    parser.add_argument("--output_dir",type=str, default=None,
                        help="Override output directory.")
    parser.add_argument("--retriever", type=str, default=None,
                        choices=["bm25", "dense"],
                        help="Override retriever method.")
    parser.add_argument("--retrievers", type=str, nargs="+", default=None,
                        choices=["bm25", "dense"],
                        help="Run one experiment matrix per retriever method.")
    parser.add_argument("--profile", type=str, default="default",
                        choices=["default", "presentation"],
                        help="Apply a reproducible experiment profile.")
    parser.add_argument("--device", type=str, default=None,
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Override runtime device.")
    parser.add_argument("--top_k",    type=int, nargs="+", default=None,
                        help="Override top-k values (e.g., --top_k 3 5).")
    parser.add_argument("--top_n",    type=int, nargs="+", default=None,
                        help="Override top-n values for context selection.")
    parser.add_argument("--no_ablations", action="store_true",
                        help="Skip ablation studies.")
    parser.add_argument("--quick", action="store_true",
                        help="Run a small smoke experiment with built-in data and no COMET.")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.config:
        cfg = ExperimentConfig.load(args.config)
        logger.info(f"Config loaded from {args.config}")
    else:
        cfg = ExperimentConfig()

    apply_profile(cfg, args.profile)

    if args.quick:
        cfg.data.test_source = "builtin"
        cfg.data.corpus_source = "builtin"
        cfg.data.test_size = 3
        cfg.retriever.methods = ["bm25"]
        cfg.retriever.method = "bm25"
        cfg.retriever.top_k_values = [1]
        cfg.context_selector.top_n_values = [1]
        cfg.evaluation.compute_comet = False
        cfg.run_ablations = False

    if args.test_size is not None:
        cfg.data.test_size = args.test_size
    if args.seed is not None:
        cfg.seed = args.seed
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.retriever is not None:
        cfg.retriever.method = args.retriever
        cfg.retriever.methods = [args.retriever]
    if args.retrievers is not None:
        cfg.retriever.methods = args.retrievers
        cfg.retriever.method = args.retrievers[0]
    if args.device is not None:
        cfg.device = args.device
    if args.top_k is not None:
        cfg.retriever.top_k_values = args.top_k
    if args.top_n is not None:
        cfg.context_selector.top_n_values = args.top_n
    if args.no_ablations:
        cfg.run_ablations = False

    logger.info("═" * 60)
    logger.info("RAG-MT Research Experiment")
    logger.info(f"  Language pair  : EN → FI")
    logger.info(f"  MT model       : {cfg.translator.model_name}")
    logger.info(f"  Retriever      : {cfg.retriever.method.upper()} (k={cfg.retriever.top_k_values})")
    logger.info(f"  Retriever runs : {cfg.retriever.methods}")
    logger.info(f"  Context scorer : {cfg.context_selector.scoring_method}")
    logger.info(f"  Top-N values   : {cfg.context_selector.top_n_values}")
    logger.info(f"  Test size      : {cfg.data.test_size}")
    logger.info(f"  Device         : {cfg.device}")
    logger.info(f"  Seed           : {cfg.seed}")
    logger.info(f"  Output dir     : {cfg.output_dir}")
    logger.info("═" * 60)

    run_all_experiments(cfg)
    logger.info("All experiments complete.")


if __name__ == "__main__":
    main()
