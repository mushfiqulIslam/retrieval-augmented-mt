import logging
import time

from utils.data import TranslationResult

logger = logging.getLogger(__name__)

def run_system_b(
        test_pairs, retriever, translator, naive_selector, top_k
):
    """
    System B — RAG Naïve Concatenation.
    Pipeline: source → retrieve top-k → concatenate full docs → MT model.
    No filtering. Same retrieved docs as System C.
    """
    logger.info(f"\n{'─'*60}")
    logger.info(f"Running System B: RAG-Naïve (k={top_k})")
    logger.info("─" * 60)

    sources    = [p["en"] for p in test_pairs]
    references = [p["fi"] for p in test_pairs]

    # Build (source, context) pairs
    contexts     = []
    ctx_tokens   = []
    for src in sources:
        retrieved = retriever.retrieve(src, top_k=top_k)
        context   = naive_selector.select(retrieved)
        contexts.append(context)
        ctx_tokens.append(translator.count_tokens(context) if context else 0)

    start = time.perf_counter()
    hypotheses = translator.translate_batch(sources, contexts=contexts)
    elapsed = time.perf_counter() - start

    logger.info(f"System B complete. {len(hypotheses)} sentences in {elapsed:.1f}s. "
                f"Avg context tokens: {sum(ctx_tokens)/len(ctx_tokens):.1f}")

    results = []
    name = f"System_B_k{top_k}"
    for src, ref, hyp, ctx, ctok in zip(sources, references, hypotheses, contexts, ctx_tokens):
        results.append(TranslationResult(
            source=src, reference=ref, hypothesis=hyp,
            context=ctx, context_tokens=ctok, system_name=name
        ))
    return results