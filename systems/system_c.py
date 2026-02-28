import logging
import time

from utils.data import TranslationResult

logger = logging.getLogger(__name__)

def run_system_c(
        test_pairs, retriever, translator,
        filtered_selector, top_k, top_n, system_suffix = "",
):
    """
    System C — RAG Filtered Context (proposed method).
    Pipeline: source → retrieve same top-k → sentence segment → score → top-N → MT.
    Only context SELECTION differs from System B.
    """
    label = f"System_C_k{top_k}_N{top_n}" + (f"_{system_suffix}" if system_suffix else "")
    logger.info(f"\n{'─'*60}")
    logger.info(f"Running {label}")
    logger.info("─" * 60)

    sources    = [p["en"] for p in test_pairs]
    references = [p["fi"] for p in test_pairs]

    contexts   = []
    ctx_tokens = []

    for src in sources:
        retrieved = retriever.retrieve(src, top_k=top_k)
        selected_sents, _ = filtered_selector.select(src, retrieved, top_n=top_n)
        context = filtered_selector.build_context_string(selected_sents)
        contexts.append(context)
        ctx_tokens.append(translator.count_tokens(context) if context else 0)

    start = time.perf_counter()
    hypotheses = translator.translate_batch(sources, contexts=contexts)
    elapsed = time.perf_counter() - start

    logger.info(f"{label} complete. {len(hypotheses)} sentences in {elapsed:.1f}s. "
                f"Avg context tokens: {sum(ctx_tokens)/len(ctx_tokens):.1f}")

    results = []
    for src, ref, hyp, ctx, ctok in zip(sources, references, hypotheses, contexts, ctx_tokens):
        results.append(TranslationResult(
            source=src, reference=ref, hypothesis=hyp,
            context=ctx, context_tokens=ctok, system_name=label
        ))
    return results


def run_system_c_random(
        test_pairs, retriever, translator, random_selector, top_k, top_n,
):
    """
    Ablation: Random Context Selection (control condition).
    Like System C but sentences chosen randomly, not by relevance score.
    """
    label = f"Ablation_Random_k{top_k}_N{top_n}"
    logger.info(f"{'─'*60}")
    logger.info(f"Running {label}")
    logger.info("─" * 60)

    sources    = [p["en"] for p in test_pairs]
    references = [p["fi"] for p in test_pairs]

    contexts   = []
    ctx_tokens = []

    for src in sources:
        retrieved = retriever.retrieve(src, top_k=top_k)
        selected  = random_selector.select(retrieved, top_n=top_n)
        context   = " ".join(selected)
        contexts.append(context)
        ctx_tokens.append(translator.count_tokens(context) if context else 0)

    start = time.perf_counter()
    hypotheses = translator.translate_batch(sources, contexts=contexts)
    elapsed = time.perf_counter() - start

    logger.info(f"{label} complete in {elapsed:.1f}s.")

    results = []
    for src, ref, hyp, ctx, ctok in zip(sources, references, hypotheses, contexts, ctx_tokens):
        results.append(TranslationResult(
            source=src, reference=ref, hypothesis=hyp,
            context=ctx, context_tokens=ctok, system_name=label
        ))
    return results