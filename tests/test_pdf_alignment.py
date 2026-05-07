import json
import os
import tempfile
import unittest
from unittest import mock

from run_experiments import apply_profile
from systems.run_all_experiments import run_all_experiments
from systems.system_c import run_system_c
from utils.config import ExperimentConfig
from utils.data import MetricScores, TranslationResult
from utils.reporting import build_hypothesis_verdict, build_length_bucket_analysis


class FakeRetriever:
    def retrieve(self, query, top_k):
        return [
            {
                "doc_id": "doc1",
                "title": "Doc 1",
                "text": "Relevant sentence one. Less relevant sentence two.",
                "score": 0.9,
                "rank": 1,
            }
        ]


class FakeTranslator:
    def __init__(self, cfg=None, device="cpu"):
        self.cfg = cfg
        self.device = device

    def translate_batch(self, sources, contexts=None):
        return [f"hyp-{i}" for i, _ in enumerate(sources)]

    def count_tokens(self, text):
        return len(text.split())


class FakeSelector:
    def select(self, source, retrieved_docs, top_n):
        scored = [("Relevant sentence one.", 0.95), ("Less relevant sentence two.", 0.2)]
        return [sent for sent, _ in scored[:top_n]], scored

    def build_context_string(self, selected_sentences):
        return " ".join(selected_sentences)


class FakeEvaluator:
    def __init__(self, *args, **kwargs):
        self._comet_eval = None
        self._hall_eval = None

    def evaluate(self, results, system_name, metadata=None):
        return MetricScores(
            system_name=system_name,
            bleu=1.0,
            comet=0.0,
            comet_metric_name="disabled",
            hallucination_rate=0.1,
            avg_context_tokens=2.0 if system_name != "System_A" else 0.0,
            quality_per_token=0.5 if system_name != "System_A" else 0.0,
            n_sentences=len(results),
            metadata=metadata or {},
        )


def fake_system_a(test_pairs, translator, cfg):
    return [
        TranslationResult(pair["en"], pair["fi"], "hyp-a", None, 0, "System_A")
        for pair in test_pairs
    ]


def fake_rag_results(test_pairs, *args, retriever_method=None, top_k=1, top_n=None, trace_records=None, **kwargs):
    system_name = f"Fake_{retriever_method}_k{top_k}" + (f"_N{top_n}" if top_n else "")
    return [
        TranslationResult(pair["en"], pair["fi"], "hyp-rag", "ctx", 1, system_name)
        for pair in test_pairs
    ]


class PdfAlignmentTests(unittest.TestCase):
    def test_presentation_profile_matches_pdf_setup(self):
        cfg = ExperimentConfig()

        apply_profile(cfg, "presentation")

        self.assertEqual(cfg.data.test_source, "hf_dataset")
        self.assertEqual(cfg.data.corpus_source, "builtin")
        self.assertEqual(cfg.retriever.top_k_values, [3, 5])
        self.assertEqual(cfg.context_selector.top_n_values, [1, 3, 5])
        self.assertEqual(cfg.translator.num_beams, 4)
        self.assertEqual(cfg.seed, 42)

    @mock.patch("systems.run_all_experiments.print_research_conclusions")
    @mock.patch("systems.run_all_experiments.print_hallucination_examples")
    @mock.patch("systems.run_all_experiments.print_results_table")
    @mock.patch("systems.run_all_experiments.MasterEvaluator", FakeEvaluator)
    @mock.patch("systems.run_all_experiments.Translator", FakeTranslator)
    @mock.patch("systems.run_all_experiments.build_retriever", return_value=FakeRetriever())
    @mock.patch("systems.run_all_experiments.run_system_c_random", fake_rag_results)
    @mock.patch("systems.run_all_experiments.run_system_c", fake_rag_results)
    @mock.patch("systems.run_all_experiments.run_system_b", fake_rag_results)
    @mock.patch("systems.run_all_experiments.run_system_a", fake_system_a)
    def test_multiple_retrievers_have_method_qualified_names_and_single_system_a(self, *_):
        cfg = ExperimentConfig()
        cfg.device = "cpu"
        cfg.data.test_source = "builtin"
        cfg.data.corpus_source = "builtin"
        cfg.data.test_size = 1
        cfg.retriever.methods = ["bm25", "dense"]
        cfg.retriever.top_k_values = [1]
        cfg.context_selector.top_n_values = [1]
        cfg.evaluation.compute_comet = False
        cfg.run_ablations = False
        cfg.save_translations = False

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            run_all_experiments(cfg)
            with open(os.path.join(tmpdir, "all_scores.json"), encoding="utf-8") as f:
                names = [row["system_name"] for row in json.load(f)]

        self.assertEqual(names.count("System_A"), 1)
        self.assertIn("System_B_bm25_k1", names)
        self.assertIn("System_C_bm25_k1_N1", names)
        self.assertIn("System_B_dense_k1", names)
        self.assertIn("System_C_dense_k1_N1", names)

    def test_system_c_context_trace_records_selected_sentence_scores(self):
        traces = []
        test_pairs = [{"en": "source", "fi": "reference"}]

        run_system_c(
            test_pairs,
            FakeRetriever(),
            FakeTranslator(),
            FakeSelector(),
            top_k=1,
            top_n=1,
            retriever_method="bm25",
            trace_records=traces,
        )

        self.assertEqual(len(traces), 1)
        trace = traces[0]
        self.assertEqual(trace["system"], "System_C_bm25_k1_N1")
        self.assertEqual(trace["retriever_method"], "bm25")
        self.assertEqual(trace["retrieved_docs"][0]["doc_id"], "doc1")
        self.assertEqual(trace["selected_sentence_scores"][0]["score"], 0.95)

    def test_report_verdict_computes_hypotheses_from_scores(self):
        scores = [
            MetricScores("System_A", bleu=20.0, hallucination_rate=0.2, avg_context_tokens=0, quality_per_token=0),
            MetricScores("System_B_bm25_k3", bleu=10.0, hallucination_rate=0.4, avg_context_tokens=100, quality_per_token=0.1),
            MetricScores("System_C_bm25_k3_N1", bleu=15.0, hallucination_rate=0.1, avg_context_tokens=10, quality_per_token=1.5),
        ]

        verdict = build_hypothesis_verdict(scores)

        self.assertEqual(verdict["h1_quality_ranking"]["ranking"], ["System_A", "System_C_bm25_k3_N1", "System_B_bm25_k3"])
        self.assertTrue(verdict["h2_entity_novelty"]["passed"])
        self.assertTrue(verdict["h3_context_efficiency"]["passed"])

    def test_length_bucket_analysis_is_deterministic(self):
        results = [
            TranslationResult("short source", "ref", "hyp", None, 0, "System_A"),
            TranslationResult(" ".join(["medium"] * 12), "ref", "hyp", "ctx", 3, "System_B_bm25_k1"),
            TranslationResult(" ".join(["long"] * 25), "ref", "hyp", "ctx", 7, "System_C_bm25_k1_N1"),
        ]
        scores_by_system = {
            "System_A": 0.1,
            "System_B_bm25_k1": 0.2,
            "System_C_bm25_k1_N1": 0.3,
        }

        analysis = build_length_bucket_analysis({"mixed": results}, entity_rates=scores_by_system)

        self.assertEqual(analysis["mixed"]["short"]["n_sentences"], 1)
        self.assertEqual(analysis["mixed"]["medium"]["n_sentences"], 1)
        self.assertEqual(analysis["mixed"]["long"]["n_sentences"], 1)
        self.assertEqual(analysis["mixed"]["long"]["avg_context_tokens"], 7.0)


if __name__ == "__main__":
    unittest.main()
