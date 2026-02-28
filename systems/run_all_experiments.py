import json
import logging
import os
import random
from typing import List, Dict

import numpy as np
import torch

from context_selector.filtered_context_selector import FilteredContextSelector
from context_selector.naive_context_selector import NaiveContextSelector
from context_selector.random_context_selector import RandomContextSelector
from evaluator.master_evaluator import MasterEvaluator
from retriever.base import BaseRetriever
from retriever.bm_25_retriever import BM25Retriever
from retriever.cached_retriever import CachedRetriever
from systems.system_a import run_system_a
from systems.system_b import run_system_b
from systems.system_c import run_system_c, run_system_c_random
from translator.translator import Translator
from utils.config import ExperimentConfig, RetrieverConfig
from utils.data import load_test_set, load_retrieval_corpus
from utils.utils import save_translations, print_results_table, print_hallucination_examples, save_scores, \
    print_research_conclusions

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


def build_retriever(corpus: List[Dict], cfg: RetrieverConfig) -> BaseRetriever:
    """
    The returned object is frozen (index is built once and settings never change).
    """
    if cfg.method == "bm25":
        return BM25Retriever(corpus, cfg)
    elif cfg.method == "dense":
        raise ValueError("Will implement dense retriever later.")
    else:
        raise ValueError(f"Unknown retriever method: {cfg.method!r}. Choose 'bm25' or 'dense'.")


def run_all_experiments(cfg: ExperimentConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_all_seeds(cfg.seed)
    cfg.save(os.path.join(cfg.output_dir, "experiment_config.json"))
    logger.info(f"Config saved to {cfg.output_dir}/experiment_config.json")

    logger.info("═════════════════ LOADING DATA ═════════════════")
    test_pairs = load_test_set(cfg.data, seed=cfg.seed)
    corpus = load_retrieval_corpus(cfg.data)
    logger.info(f"Test sentences: {len(test_pairs)} | Corpus documents: {len(corpus)}")

    logger.info("═════════════════ BUILDING RETRIEVER ═════════════════")
    base_retriever = build_retriever(corpus, cfg.retriever)
    retriever = CachedRetriever(base_retriever)

    logger.info("═════════════════ LOADING TRANSLATOR ═════════════════")
    translator = Translator(cfg.translator)

    naive_selector = NaiveContextSelector(max_tokens=cfg.context_selector.max_context_tokens)
    filtered_selector = FilteredContextSelector(cfg.context_selector)
    random_selector = RandomContextSelector(cfg.context_selector, seed=cfg.seed)

    logger.info("═════════════════ INITIALIZING EVALUATOR ═════════════════")
    evaluator = MasterEvaluator(
        compute_bleu=cfg.evaluation.compute_bleu,
        compute_comet=cfg.evaluation.compute_comet,
        compute_hallucination=cfg.evaluation.compute_hallucination,
        compute_efficiency=cfg.evaluation.compute_context_efficiency,
        comet_model=cfg.evaluation.comet_model,
        spacy_model=cfg.evaluation.spacy_model,
    )

    all_scores = []
    all_results = {}
    logger.info("═════════════════ RUNNING EXPERIMENTS ═════════════════")

    if cfg.run_system_a:
        results_a = run_system_a(test_pairs, translator, cfg)
        all_results["System_A"] = results_a
        scores_a = evaluator.evaluate(
            results_a, "System_A",
            metadata={"system": "MT-Only"}
        )
        all_scores.append(scores_a)
        if cfg.save_translations:
            save_translations(results_a, os.path.join(cfg.output_dir, "translations_system_a.jsonl"))

    for top_k in cfg.retriever.top_k_values:
        logger.info(f"\n{'═' * 60}")
        logger.info(f"Top-k = {top_k}")
        logger.info("═" * 60)

        if cfg.run_system_b:
            results_b = run_system_b(
                test_pairs, retriever, translator, naive_selector, top_k
            )
            name_b = f"System_B_k{top_k}"
            all_results[name_b] = results_b
            scores_b = evaluator.evaluate(
                results_b, name_b,
                metadata={"system": "RAG-Naïve", "top_k": top_k}
            )
            all_scores.append(scores_b)
            if cfg.save_translations:
                save_translations(results_b, os.path.join(cfg.output_dir, f"translations_{name_b}.jsonl"))

        if cfg.run_system_c:
            for top_n in cfg.context_selector.top_n_values:
                results_c = run_system_c(
                    test_pairs, retriever, translator, filtered_selector,
                    top_k=top_k, top_n=top_n
                )
                name_c = f"System_C_k{top_k}_N{top_n}"
                all_results[name_c] = results_c
                scores_c = evaluator.evaluate(
                    results_c, name_c,
                    metadata={"system": "RAG-Filtered", "top_k": top_k, "top_n": top_n}
                )
                all_scores.append(scores_c)
                if cfg.save_translations:
                    save_translations(
                        results_c, os.path.join(cfg.output_dir, f"translations_{name_c}.jsonl")
                    )

        # Ablations
        if cfg.run_ablations:
            logger.info(f"\n{'─' * 50}  ABLATIONS (k={top_k})")
            top_n_abl = 3
            results_rand = run_system_c_random(
                test_pairs, retriever, translator, random_selector,
                top_k=top_k, top_n=top_n_abl
            )
            name_rand = f"Ablation_Random_k{top_k}_N{top_n_abl}"
            all_results[name_rand] = results_rand
            scores_rand = evaluator.evaluate(
                results_rand, name_rand,
                metadata={"system": "Ablation-Random", "top_k": top_k, "top_n": top_n_abl}
            )
            all_scores.append(scores_rand)
            if cfg.save_translations:
                save_translations(results_rand, os.path.join(cfg.output_dir,
                                                             f"translations_{name_rand}.jsonl"))

    comet_name = evaluator._comet_eval.metric_name.upper() if evaluator._comet_eval else "chrF"
    print_results_table(all_scores, comet_name=comet_name)

    if evaluator._hall_eval:
        for name, results in all_results.items():
            print_hallucination_examples(results, evaluator, name, n_examples=2)

    print("═" * 60)
    print("  CONTEXT EFFICIENCY ANALYSIS")
    print("═" * 60)
    print(f"  {'System':<35} {'BLEU':>8} {'Avg Ctx Tok':>12} {'BLEU/Tok':>10}")
    print("  " + "─" * 67)
    for s in all_scores:
        print(f"  {s.system_name:<35} {s.bleu:>8.2f} {s.avg_context_tokens:>12.1f} {s.quality_per_token:>10.4f}")
    print("═" * 60)

    scores_path = os.path.join(cfg.output_dir, "all_scores.json")
    save_scores(all_scores, scores_path)
    logger.info(f"\nAll scores saved to {scores_path}")

    hall_stats = {
        name: {
            "avg_hallucination_rate": next(
                (s.hallucination_rate for s in all_scores if s.system_name == name), None
            )
        }
        for name in all_results
    }
    hall_path = os.path.join(cfg.output_dir, "hallucination_stats.json")
    with open(hall_path, "w") as f:
        json.dump(hall_stats, f, indent=2)
    logger.info(f"Hallucination statistics saved to {hall_path}")

    print_research_conclusions(all_scores)

    return all_scores, all_results
