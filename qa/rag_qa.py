"""Retrieval-Augmented Question Answering over Finnish-culture domain documents.

This module implements a complete RAG-QA pipeline:
1. Retrieve relevant documents using BM25 or Dense retrieval
2. Extract the most relevant context passages
3. Answer questions using a generative QA model
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from retriever.bm_25_retriever import BM25Retriever
from retriever.dense_retriever import DenseRetriever
from utils.config import RetrieverConfig
from utils.sample_corpus import BUILTIN_CORPUS

logger = logging.getLogger(__name__)


@dataclass
class QAAnswer:
    question: str
    answer: str
    retrieved_docs: List[Dict]
    context_used: str
    retrieval_method: str
    model_used: str


class RAGQuestionAnswerer:
    """RAG-based QA system over domain documents.

    Retrieves relevant context and answers questions using either:
    - Extractive QA (deepset/roberta-base-squad2)
    - Generative QA (google/flan-t5-small)
    - Context-only (no model, just returns retrieved context)
    """

    def __init__(
        self,
        corpus: Optional[List[Dict]] = None,
        retriever_method: str = "bm25",
        qa_mode: str = "generative",
        device: str = "cpu",
    ):
        self.corpus = corpus or BUILTIN_CORPUS
        self.retriever_method = retriever_method
        self.qa_mode = qa_mode
        self.device = device

        ret_cfg = RetrieverConfig(method=retriever_method)
        if retriever_method == "bm25":
            self.retriever = BM25Retriever(self.corpus, ret_cfg, device)
        else:
            self.retriever = DenseRetriever(self.corpus, ret_cfg, device)

        self.qa_model = None
        self.qa_tokenizer = None
        self._load_qa_model()

        logger.info(
            "RAG-QA initialized: retriever=%s, mode=%s, docs=%d",
            retriever_method, qa_mode, len(self.corpus),
        )

    def _load_qa_model(self):
        """Load the appropriate QA model based on mode."""
        if self.qa_mode == "generative":
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                model_name = "google/flan-t5-small"
                logger.info("Loading generative QA model: %s", model_name)
                self.qa_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.qa_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.qa_model.to(self.device)
                logger.info("Generative QA model loaded.")
            except Exception as e:
                logger.warning("Failed to load generative model: %s. Falling back to extractive.", e)
                self.qa_mode = "extractive"
                self._load_qa_model()

        elif self.qa_mode == "extractive":
            try:
                from transformers import AutoModelForQuestionAnswering, AutoTokenizer
                model_name = "deepset/roberta-base-squad2"
                logger.info("Loading extractive QA model: %s", model_name)
                self.qa_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
                self.qa_model.to(self.device)
                logger.info("Extractive QA model loaded.")
            except Exception as e:
                logger.warning("Failed to load extractive model: %s. Using context-only mode.", e)
                self.qa_mode = "context"
                self.qa_model = None
                self.qa_tokenizer = None

        elif self.qa_mode == "context":
            logger.info("QA mode set to context-only (no answer generation model).")
            self.qa_model = None
            self.qa_tokenizer = None

    def answer(self, question: str, top_k: int = 3, context_sentences: int = 3) -> QAAnswer:
        """Answer a question using RAG pipeline.

        Args:
            question: The question to answer.
            top_k: Number of documents to retrieve.
            context_sentences: Number of most relevant sentences to use as context.

        Returns:
            QAAnswer with question, answer, and metadata.
        """
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        context = self._build_context(question, retrieved, context_sentences)

        if self.qa_model is not None and self.qa_tokenizer is not None:
            if self.qa_mode == "generative":
                answer_text = self._answer_generative(question, context)
            else:
                answer_text = self._answer_extractive(question, context)
        else:
            answer_text = self._answer_context_only(question, context)

        return QAAnswer(
            question=question,
            answer=answer_text,
            retrieved_docs=retrieved,
            context_used=context,
            retrieval_method=self.retriever_method,
            model_used=self.qa_mode,
        )

    def _build_context(self, question: str, docs: List[Dict], n_sentences: int) -> str:
        """Build context by selecting the most relevant sentences from retrieved docs."""
        import re

        all_sentences = []
        for doc in docs:
            sentences = re.split(r'(?<=[.!?])\s+', doc["text"])
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 10:
                    all_sentences.append(sent)

        if len(all_sentences) <= n_sentences:
            return " ".join(all_sentences)

        question_words = set(question.lower().split())

        def relevance(sentence: str) -> float:
            sent_words = set(sentence.lower().split())
            overlap = len(question_words & sent_words)
            return overlap / max(len(question_words), 1)

        scored = sorted(all_sentences, key=relevance, reverse=True)
        return " ".join(scored[:n_sentences])

    def _answer_generative(self, question: str, context: str) -> str:
        """Generate an answer using a seq2seq model (Flan-T5)."""
        import torch
        prompt = (
            f"Answer the following question based on the provided context. "
            f"Only use information from the context. Be concise.\n\n"
            f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        )
        inputs = self.qa_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.qa_model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=3,
                early_stopping=True,
            )

        answer = self.qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()

    def _answer_extractive(self, question: str, context: str) -> str:
        """Extract an answer span from context using a QA model (RoBERTa-SQuAD2)."""
        import torch
        inputs = self.qa_tokenizer(
            question, context,
            return_tensors="pt", truncation=True, max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.qa_model(**inputs)

        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits) + 1

        if start_idx >= end_idx:
            return "Could not extract a specific answer from the context."

        answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
        answer = self.qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)
        return answer.strip()

    def _answer_context_only(self, question: str, context: str) -> str:
        """Return retrieved context as the 'answer' when no QA model is available."""
        return f"[RETRIEVED CONTEXT]\n{context}\n\n[NOTE: No QA model loaded. Install transformers for answer generation.]"

    def evaluate(self, test_questions: List[Dict], top_k: int = 3) -> Dict:
        """Evaluate QA performance on test questions.

        Returns dict with accuracy, per-question results, and retrieval stats.
        """
        results = []
        correct = 0
        total = len(test_questions)

        for q in test_questions:
            qa_result = self.answer(q["question"], top_k=top_k)
            answer_lower = qa_result.answer.lower()

            is_correct = any(
                keyword.lower() in answer_lower
                for keyword in q.get("answer_contains", [])
            )

            retrieved_ids = [d.get("doc_id", "") for d in qa_result.retrieved_docs]
            target_retrieved = q.get("doc_id", "") in retrieved_ids

            results.append({
                "question": q["question"],
                "answer": qa_result.answer,
                "expected_contains": q.get("answer_contains", []),
                "correct": is_correct,
                "target_doc_retrieved": target_retrieved,
                "top_retrieved_docs": retrieved_ids[:3],
            })

            if is_correct:
                correct += 1

        retrieval_recall = sum(
            1 for r in results if r["target_doc_retrieved"]
        ) / max(total, 1)

        return {
            "total_questions": total,
            "correct_answers": correct,
            "answer_accuracy": correct / max(total, 1),
            "retrieval_recall": retrieval_recall,
            "per_question": results,
        }

    def interactive(self):
        """Run an interactive Q&A session."""
        print("\n" + "=" * 60)
        print("  RAG Question Answering — Finnish Culture Domain")
        print("  Type 'quit' or 'exit' to stop.")
        print("=" * 60 + "\n")

        while True:
            try:
                question = input("Your question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if not question:
                continue

            print(f"\nRetrieving context using {self.retriever_method}...")
            result = self.answer(question)

            print(f"\n📚 Retrieved Documents:")
            for i, doc in enumerate(result.retrieved_docs, 1):
                print(f"   {i}. [{doc['doc_id']}] {doc['title']}")

            print(f"\n📝 Context Used:\n   {result.context_used[:300]}...")
            print(f"\n💡 Answer:\n   {result.answer}")
            print(f"\n{'-' * 60}\n")
