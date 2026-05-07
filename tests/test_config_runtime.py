import json
import os
import tempfile
import unittest
from unittest import mock

from utils.config import ExperimentConfig
from utils.runtime import resolve_device


class ConfigRuntimeTests(unittest.TestCase):
    def test_config_load_restores_top_level_fields(self):
        cfg = ExperimentConfig()
        cfg.seed = 7
        cfg.output_dir = "custom-results"
        cfg.device = "cpu"
        cfg.save_translations = False
        cfg.save_scores = False
        cfg.run_system_b = False
        cfg.run_ablations = False
        cfg.data.test_source = "builtin"
        cfg.retriever.top_k_values = [1]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            cfg.save(path)

            loaded = ExperimentConfig.load(path)

        self.assertEqual(loaded.seed, 7)
        self.assertEqual(loaded.output_dir, "custom-results")
        self.assertEqual(loaded.device, "cpu")
        self.assertFalse(loaded.save_translations)
        self.assertFalse(loaded.save_scores)
        self.assertFalse(loaded.run_system_b)
        self.assertFalse(loaded.run_ablations)
        self.assertEqual(loaded.data.test_source, "builtin")
        self.assertEqual(loaded.retriever.top_k_values, [1])

    @mock.patch("utils.runtime.torch")
    def test_auto_device_prefers_cuda_then_mps_then_cpu(self, torch_mock):
        torch_mock.cuda.is_available.return_value = True
        self.assertEqual(resolve_device("auto"), "cuda")

        torch_mock.cuda.is_available.return_value = False
        torch_mock.backends.mps.is_available.return_value = True
        self.assertEqual(resolve_device("auto"), "mps")

        torch_mock.backends.mps.is_available.return_value = False
        self.assertEqual(resolve_device("auto"), "cpu")

    @mock.patch("utils.runtime.torch")
    def test_explicit_unavailable_devices_fail(self, torch_mock):
        torch_mock.cuda.is_available.return_value = False
        torch_mock.backends.mps.is_available.return_value = False

        with self.assertRaises(ValueError):
            resolve_device("cuda")
        with self.assertRaises(ValueError):
            resolve_device("mps")
        with self.assertRaises(ValueError):
            resolve_device("tpu")

    def test_macos_requirements_exclude_cuda_runtime_packages(self):
        with open("requirements-macos.txt", encoding="utf-8") as f:
            macos_requirements = f.read()
        with open("requirements.txt", encoding="utf-8") as f:
            linux_requirements = f.read()

        self.assertNotIn("cuda-", macos_requirements)
        self.assertNotIn("nvidia-", macos_requirements)
        self.assertIn("cuda-bindings", linux_requirements)
        self.assertIn("nvidia-cublas-cu12", linux_requirements)

    def test_saved_config_is_full_json_object(self):
        cfg = ExperimentConfig()
        cfg.device = "cpu"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            cfg.save(path)
            with open(path, encoding="utf-8") as f:
                saved = json.load(f)

        self.assertEqual(saved["device"], "cpu")
        self.assertIn("retriever", saved)
        self.assertIn("context_selector", saved)

    def test_old_config_without_retriever_methods_uses_method(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "old_config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"retriever": {"method": "dense"}}, f)

            loaded = ExperimentConfig.load(path)

        self.assertEqual(loaded.retriever.method, "dense")
        self.assertEqual(loaded.retriever.methods, ["dense"])


if __name__ == "__main__":
    unittest.main()
