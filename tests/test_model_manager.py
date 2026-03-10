import unittest
from unittest.mock import patch

from chatbot.model_manager import (
    _build_load_candidates,
    _recommend_gpu_layers,
    _recommended_contexts,
)


class TestModelManagerHeuristics(unittest.TestCase):
    def setUp(self):
        self.dual_3060 = [
            {"index": 0, "memory_total_gb": 12.0, "memory_free_gb": 11.9, "name": "RTX 3060"},
            {"index": 1, "memory_total_gb": 12.0, "memory_free_gb": 11.0, "name": "RTX 3060"},
        ]

    def test_large_model_contexts_include_downshift_steps(self):
        contexts = _recommended_contexts(requested_n_ctx=8192, file_size_gb=20.5, trained_ctx=262144)
        self.assertEqual(contexts, [8192, 6144, 4096, 3072, 2048])

    def test_gpu_layer_recommendation_uses_most_layers_when_it_nearly_fits(self):
        layers = _recommend_gpu_layers(
            file_size_gb=19.85,
            total_layers=64,
            gpus=self.dual_3060,
            n_ctx=4096,
            requested_n_gpu_layers=-1,
        )
        self.assertGreaterEqual(layers, 60)

    @patch("chatbot.model_manager._llama_gpu_offload_enabled", return_value=True)
    def test_build_load_candidates_prefers_gpu_then_cpu_fallback(self, _mock_gpu_support):
        candidates = _build_load_candidates(
            file_size_gb=20.5,
            total_layers=40,
            gpus=self.dual_3060,
            requested_n_ctx=8192,
            requested_n_gpu_layers=-1,
            trained_ctx=262144,
        )

        self.assertGreaterEqual(len(candidates), 4)
        self.assertNotEqual(candidates[0]["n_gpu_layers"], 0)
        self.assertEqual(candidates[-1]["n_gpu_layers"], 0)
        self.assertIsNotNone(candidates[0]["tensor_split"])


if __name__ == "__main__":
    unittest.main()
