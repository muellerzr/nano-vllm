import unittest

from benchmarks.experiment_config import build_compilation_config


class ExperimentConfigTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "mode": 3,
            "cudagraph_mode": "NONE",
            "pass_config": {"enable_qk_norm_rope_fusion": True},
        }

    def test_off_is_deep_equal_and_does_not_mutate(self):
        original = {
            "mode": 3,
            "cudagraph_mode": "NONE",
            "pass_config": {"enable_qk_norm_rope_fusion": True},
        }
        result = build_compilation_config(self.base)
        self.assertEqual(result, original)
        self.assertIsNot(result, self.base)
        self.assertIsNot(result["pass_config"], self.base["pass_config"])

    def test_overlap_enables_only_native_async_tp_passes(self):
        result = build_compilation_config(self.base, overlap=True, sp_min_token_num=1)
        self.assertTrue(result["pass_config"]["enable_sp"])
        self.assertTrue(result["pass_config"]["fuse_gemm_comms"])
        self.assertEqual(result["pass_config"]["sp_min_token_num"], 1)
        self.assertTrue(result["pass_config"]["enable_qk_norm_rope_fusion"])

    def test_overlap_without_threshold_preserves_vllm_heuristic(self):
        result = build_compilation_config(self.base, overlap=True)
        self.assertNotIn("sp_min_token_num", result["pass_config"])

    def test_sharded_downstream_enables_native_allreduce_rms(self):
        result = build_compilation_config(self.base, sharded_downstream=True)
        self.assertTrue(result["pass_config"]["fuse_allreduce_rms"])


if __name__ == "__main__":
    unittest.main()
