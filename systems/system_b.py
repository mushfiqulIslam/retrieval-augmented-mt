import logging
import time

from utils.data import TranslationResult

logger = logging.getLogger(__name__)

def run_system_b(
        test_pairs, retriever, translator, naive_selector, top_k,
        retriever_method=None, trace_records=None,
):
    """
    System B — RAG Naïve Concatenation.
    Pipeline: source → retrieve top-k → concatenate full docs → MT model.
    No filtering. Same retrieved docs as System C.
    """
    logger.info(f"\n{'─'*60}")
    method_label = f"_{retriever_method}" if retriever_method else ""
    name = f"System_B{method_label}_k{top_k}"
    logger.info(f"Running System B: RAG-Naïve ({retriever_method or 'retriever'}, k={top_k})")
    logger.info("─" * 60)

    sources    = [p["en"] for p in test_pairs]
    references = [p["fi"] for p in test_pairs]

    # Build (source, context) pairs
    contexts     = []
    ctx_tokens   = []
    retrieved_by_source = []
    for src in sources:
        retrieved = retriever.retrieve(src, top_k=top_k)
        context   = naive_selector.select(retrieved)
        contexts.append(context)
        ctx_tokens.append(translator.count_tokens(context) if context else 0)
        retrieved_by_source.append(retrieved)

    start = time.perf_counter()
    hypotheses = translator.translate_batch(sources, contexts=contexts)
    elapsed = time.perf_counter() - start

    logger.info(f"System B complete. {len(hypotheses)} sentences in {elapsed:.1f}s. "
                f"Avg context tokens: {sum(ctx_tokens)/len(ctx_tokens):.1f}")

    results = []
    for src, ref, hyp, ctx, ctok, retrieved in zip(sources, references, hypotheses, contexts, ctx_tokens, retrieved_by_source):
        results.append(TranslationResult(
            source=src, reference=ref, hypothesis=hyp,
            context=ctx, context_tokens=ctok, system_name=name
        ))
        if trace_records is not None:
            trace_records.append({
                "system": name,
                "retriever_method": retriever_method,
                "top_k": top_k,
                "top_n": None,
                "source": src,
                "reference": ref,
                "hypothesis": hyp,
                "context_tokens": ctok,
                "retrieved_docs": _trace_docs(retrieved),
                "selected_sentences": [],
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
