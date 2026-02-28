import logging
import time

from utils.data import TranslationResult

logger = logging.getLogger(__name__)

def run_system_a(test_pairs, translator, cfg):
    logger.info("\n" + "─" * 60)
    logger.info("Running System A: MT-Only Baseline")
    logger.info("─" * 60)

    sources = [p["en"] for p in test_pairs]
    references = [p["fi"] for p in test_pairs]

    start = time.perf_counter()
    hypotheses = translator.translate_batch(sources, contexts=None)
    elapsed = time.perf_counter() - start

    logger.info(f"System A complete. {len(hypotheses)} sentences translated in {elapsed:.1f}s.")

    results = []
    for src, ref, hyp in zip(sources, references, hypotheses):
        results.append(TranslationResult(
            source=src, reference=ref, hypothesis=hyp,
            context=None, context_tokens=0, system_name="System_A"
        ))
    return results