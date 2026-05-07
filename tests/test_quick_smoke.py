import json
import os
import tempfile
import unittest
from unittest import mock

from systems.run_all_experiments import run_all_experiments
from utils.config import ExperimentConfig
from utils.data import MetricScores, TranslationResult


class FakeTranslator:
    def __init__(self, cfg, device="cpu"):
        self.cfg = cfg
        self.device = device

    def translate_batch(self, sources, contexts=None):
        return ["käännös" for _ in sources]

    def count_tokens(self, text):
        return len(text.split())


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
            hallucination_rate=0.0,
            avg_context_tokens=0.0,
            quality_per_token=0.0,
            n_sentences=len(results),
            metadata=metadata or {},
        )


def fake_system_results(test_pairs, *args, **kwargs):
    return [
        TranslationResult(
            source=pair["en"],
            reference=pair["fi"],
            hypothesis="käännös",
            context=None,
            context_tokens=0,
            system_name="Fake",
        )
        for pair in test_pairs
    ]


class QuickSmokeTests(unittest.TestCase):
    @mock.patch("systems.run_all_experiments.print_research_conclusions")
    @mock.patch("systems.run_all_experiments.print_hallucination_examples")
    @mock.patch("systems.run_all_experiments.print_results_table")
    @mock.patch("systems.run_all_experiments.MasterEvaluator", FakeEvaluator)
    @mock.patch("systems.run_all_experiments.Translator", FakeTranslator)
    @mock.patch("systems.run_all_experiments.run_system_c_random", fake_system_results)
    @mock.patch("systems.run_all_experiments.run_system_c", fake_system_results)
    @mock.patch("systems.run_all_experiments.run_system_b", fake_system_results)
    @mock.patch("systems.run_all_experiments.run_system_a", fake_system_results)
    def test_quick_style_run_writes_metadata_and_scores(self, *_):
        cfg = ExperimentConfig()
        cfg.device = "cpu"
        cfg.data.test_source = "builtin"
        cfg.data.corpus_source = "builtin"
        cfg.data.test_size = 2
        cfg.retriever.top_k_values = [1]
        cfg.context_selector.top_n_values = [1]
        cfg.evaluation.compute_comet = False
        cfg.run_ablations = False

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            run_all_experiments(cfg)

            expected_files = {
                "experiment_config.json",
                "run_metadata.json",
                "all_scores.json",
                "entity_novelty_stats.json",
            }
            self.assertTrue(expected_files.issubset(set(os.listdir(tmpdir))))

            with open(os.path.join(tmpdir, "run_metadata.json"), encoding="utf-8") as f:
                metadata = json.load(f)
            self.assertEqual(metadata["requested_device"], "cpu")
            self.assertEqual(metadata["resolved_device"], "cpu")
            self.assertEqual(metadata["models"]["translator"], cfg.translator.model_name)

            with open(os.path.join(tmpdir, "entity_novelty_stats.json"), encoding="utf-8") as f:
                entity_stats = json.load(f)
            self.assertIn("Lightweight entity novelty heuristic", entity_stats["_metric_note"])
            self.assertIn("systems", entity_stats)


if __name__ == "__main__":
    unittest.main()
