#!/usr/bin/env python3
"""Run RAG-based Question Answering over Finnish-culture domain documents.

This demonstrates the RAG approach for domain-specific QA:
- Retrieves relevant documents using BM25 or dense retrieval
- Answers questions using a generative (Flan-T5) or extractive (RoBERTa) model
- Evaluates QA accuracy on test questions

Usage:
    python run_qa.py                          # Evaluate all test questions
    python run_qa.py --interactive            # Interactive Q&A session
    python run_qa.py --question "..."         # Single question
    python run_qa.py --retriever dense        # Use dense retrieval
    python run_qa.py --mode context           # No model, just retrieval
"""

from __future__ import annotations

import argparse
import logging

from qa.rag_qa import RAGQuestionAnswerer
from qa.questions import QA_TEST_QUESTIONS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_qa")


def main():
    parser = argparse.ArgumentParser(
        description="RAG Question Answering over Finnish-culture domain"
    )
    parser.add_argument(
        "--retriever", choices=["bm25", "dense"], default="bm25",
        help="Retrieval method (default: bm25)",
    )
    parser.add_argument(
        "--mode", choices=["generative", "extractive", "context"], default="generative",
        help="QA mode: generative (Flan-T5), extractive (RoBERTa), or context-only",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of documents to retrieve (default: 3)",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Start interactive Q&A session",
    )
    parser.add_argument(
        "--question", type=str,
        help="Answer a single question",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device to run QA model on (cpu/cuda/mps)",
    )

    args = parser.parse_args()

    logger.info("Initializing RAG-QA system...")
    qa = RAGQuestionAnswerer(
        retriever_method=args.retriever,
        qa_mode=args.mode,
        device=args.device,
    )

    if args.interactive:
        qa.interactive()
        return

    if args.question:
        result = qa.answer(args.question, top_k=args.top_k)

        print(f"\n{'=' * 60}")
        print(f"  Q: {result.question}")
        print(f"{'=' * 60}")
        print(f"\n📚 Retrieved Documents:")
        for i, doc in enumerate(result.retrieved_docs, 1):
            print(f"   {i}. [{doc['doc_id']}] {doc['title']}")
        print(f"\n📝 Context Used:")
        for line in result.context_used.split(". "):
            if line.strip():
                print(f"   • {line.strip()}")
        print(f"\n💡 Answer: {result.answer}")
        print(f"\n{'=' * 60}")
        return

    logger.info(
        "Evaluating on %d test questions (retriever=%s, mode=%s)...",
        len(QA_TEST_QUESTIONS), args.retriever, args.mode,
    )
    metrics = qa.evaluate(QA_TEST_QUESTIONS, top_k=args.top_k)

    print(f"\n{'=' * 60}")
    print(f"  RAG-QA Evaluation Results")
    print(f"  Retriever: {args.retriever} | Mode: {args.mode} | Top-k: {args.top_k}")
    print(f"{'=' * 60}")
    print(f"\n  Answer Accuracy:    {metrics['answer_accuracy']:.1%} ({metrics['correct_answers']}/{metrics['total_questions']})")
    print(f"  Retrieval Recall:   {metrics['retrieval_recall']:.1%}")
    print(f"\n  Per-Question Results:")
    print(f"  {'─' * 54}")

    for r in metrics["per_question"]:
        status = "✓" if r["correct"] else "✗"
        print(f"  {status} {r['question']}")
        print(f"    Answer: {r['answer'][:100]}{'...' if len(r['answer']) > 100 else ''}")
        print(f"    Retrieved: {r['top_retrieved_docs']} | Target doc: {'✓' if r['target_doc_retrieved'] else '✗'}")
        if not r["correct"]:
            print(f"    Expected keywords: {r['expected_contains']}")

    print(f"\n  {'─' * 54}")
    print(f"  Summary: {metrics['correct_answers']}/{metrics['total_questions']} correct")
    print(f"  Answer accuracy: {metrics['answer_accuracy']:.1%}")
    print(f"  Retrieval recall: {metrics['retrieval_recall']:.1%}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
