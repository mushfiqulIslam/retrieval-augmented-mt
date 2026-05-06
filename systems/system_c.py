import logging
import time

from utils.data import TranslationResult

logger = logging.getLogger(__name__)

def run_system_c(
        test_pairs, retriever, translator,
        filtered_selector, top_k, top_n, system_suffix = "",
        retriever_method=None, trace_records=None,
):
    """
    System C — RAG Filtered Context (proposed method).
    Pipeline: source → retrieve same top-k → sentence segment → score → top-N → MT.
    Only context SELECTION differs from System B.
    """
    method_label = f"_{retriever_method}" if retriever_method else ""
    label = f"System_C{method_label}_k{top_k}_N{top_n}" + (f"_{system_suffix}" if system_suffix else "")
    logger.info(f"\n{'─'*60}")
    logger.info(f"Running {label}")
    logger.info("─" * 60)

    sources    = [p["en"] for p in test_pairs]
    references = [p["fi"] for p in test_pairs]

    contexts   = []
    ctx_tokens = []
    retrieved_by_source = []
    selected_by_source = []
    scored_by_source = []

    for src in sources:
        retrieved = retriever.retrieve(src, top_k=top_k)
        selected_sents, scored = filtered_selector.select(src, retrieved, top_n=top_n)
        context = filtered_selector.build_context_string(selected_sents)
        contexts.append(context)
        ctx_tokens.append(translator.count_tokens(context) if context else 0)
        retrieved_by_source.append(retrieved)
        selected_by_source.append(selected_sents)
        scored_by_source.append(scored)

    start = time.perf_counter()
    hypotheses = translator.translate_batch(sources, contexts=contexts)
    elapsed = time.perf_counter() - start

    logger.info(f"{label} complete. {len(hypotheses)} sentences in {elapsed:.1f}s. "
                f"Avg context tokens: {sum(ctx_tokens)/len(ctx_tokens):.1f}")

    results = []
    for src, ref, hyp, ctx, ctok, retrieved, selected, scored in zip(
        sources, references, hypotheses, contexts, ctx_tokens,
        retrieved_by_source, selected_by_source, scored_by_source
    ):
        results.append(TranslationResult(
            source=src, reference=ref, hypothesis=hyp,
            context=ctx, context_tokens=ctok, system_name=label
        ))
        if trace_records is not None:
            trace_records.append({
                "system": label,
                "retriever_method": retriever_method,
                "top_k": top_k,
                "top_n": top_n,
                "source": src,
                "reference": ref,
                "hypothesis": hyp,
                "context_tokens": ctok,
                "retrieved_docs": _trace_docs(retrieved),
                "selected_sentences": selected,
                "selected_sentence_scores": [
                    {"sentence": sentence, "score": float(score)}
                    for sentence, score in scored[:top_n]
                ],
                "candidate_sentence_scores": [
                    {"sentence": sentence, "score": float(score)}
                    for sentence, score in scored
                ],
            })
    return results


def run_system_c_random(
        test_pairs, retriever, translator, random_selector, top_k, top_n,
        retriever_method=None, trace_records=None,
):
    """
    Ablation: Random Context Selection (control condition).
    Like System C but sentences chosen randomly, not by relevance score.
    """
    method_label = f"_{retriever_method}" if retriever_method else ""
    label = f"Ablation_Random{method_label}_k{top_k}_N{top_n}"
    logger.info(f"{'─'*60}")
    logger.info(f"Running {label}")
    logger.info("─" * 60)

    sources    = [p["en"] for p in test_pairs]
    references = [p["fi"] for p in test_pairs]

    contexts   = []
    ctx_tokens = []
    retrieved_by_source = []
    selected_by_source = []

    for src in sources:
        retrieved = retriever.retrieve(src, top_k=top_k)
        selected  = random_selector.select(retrieved, top_n=top_n)
        context   = " ".join(selected)
        contexts.append(context)
        ctx_tokens.append(translator.count_tokens(context) if context else 0)
        retrieved_by_source.append(retrieved)
        selected_by_source.append(selected)

    start = time.perf_counter()
    hypotheses = translator.translate_batch(sources, contexts=contexts)
    elapsed = time.perf_counter() - start

    logger.info(f"{label} complete in {elapsed:.1f}s.")

    results = []
    for src, ref, hyp, ctx, ctok, retrieved, selected in zip(
        sources, references, hypotheses, contexts, ctx_tokens, retrieved_by_source, selected_by_source
    ):
        results.append(TranslationResult(
            source=src, reference=ref, hypothesis=hyp,
            context=ctx, context_tokens=ctok, system_name=label
        ))
        if trace_records is not None:
            trace_records.append({
                "system": label,
                "retriever_method": retriever_method,
                "top_k": top_k,
                "top_n": top_n,
                "source": src,
                "reference": ref,
                "hypothesis": hyp,
                "context_tokens": ctok,
                "retrieved_docs": _trace_docs(retrieved),
                "selected_sentences": selected,
                "selected_sentence_scores": [],
            })
    return results


def _trace_docs(retrieved_docs):
    return [
        {
            "doc_id": doc.get("doc_id", doc.get("id")),
            "title": doc.get("title", ""),
            "score": float(doc["score"]) if doc.get("score") is not None else None,
            "rank": doc.get("rank"),
        }
        for doc in retrieved_docs
    ]
