import unittest

from benchmarks.feature_policy import (
    native_overlap_case_decision,
    native_overlap_decision,
    sharded_downstream_decision,
)


class FeaturePolicyTests(unittest.TestCase):
    def test_native_overlap_rejects_minimax_hidden_size(self):
        selected, reason, threshold = native_overlap_decision(
            requested=True,
            capability_major=12,
            hidden_size=3072,
            tensor_parallel_size=4,
            tokens=2048,
        )
        self.assertFalse(selected)
        self.assertEqual(reason, "hidden_size")
        self.assertIsNone(threshold)

    def test_native_overlap_force_bypasses_model_size_threshold(self):
        selected, reason, threshold = native_overlap_decision(
            requested=True,
            force=True,
            capability_major=12,
            hidden_size=3072,
            tensor_parallel_size=4,
            tokens=1,
        )
        self.assertTrue(selected)
        self.assertEqual(reason, "forced")
        self.assertIsNone(threshold)

    def test_native_overlap_uses_size_threshold_for_eligible_model(self):
        selected, reason, threshold = native_overlap_decision(
            requested=True,
            capability_major=12,
            hidden_size=8192,
            tensor_parallel_size=4,
            tokens=8191,
        )
        self.assertFalse(selected)
        self.assertEqual(reason, "token_threshold")
        self.assertEqual(threshold, 8192)

    def test_native_overlap_requires_isolated_stable_shape(self):
        selected, reason = native_overlap_case_decision(
            selected=True,
            isolated_case=False,
            phase="prefill",
            batch=32,
            context=16,
        )
        self.assertFalse(selected)
        self.assertEqual(reason, "isolated_case_required")
        selected, reason = native_overlap_case_decision(
            selected=True,
            isolated_case=True,
            phase="prefill",
            batch=32,
            context=16,
        )
        self.assertTrue(selected)
        self.assertEqual(reason, "isolated_shape")
        selected, reason = native_overlap_case_decision(
            selected=True,
            isolated_case=True,
            phase="prefill",
            batch=4,
            context=512,
        )
        self.assertFalse(selected)
        self.assertEqual(reason, "shape_repeat_unsupported")
        selected, reason, threshold = native_overlap_decision(
            requested=True,
            capability_major=12,
            hidden_size=8192,
            tensor_parallel_size=4,
            tokens=8192,
        )
        self.assertTrue(selected)
        self.assertEqual(reason, "eligible")
        self.assertEqual(threshold, 8192)

    def test_sharded_downstream_stays_off_for_fused_moe(self):
        selected, reason = sharded_downstream_decision(
            requested=True,
            fused_moe_requires_full_hidden=True,
            backend_available=True,
        )
        self.assertFalse(selected)
        self.assertEqual(reason, "fused_moe_requires_full_hidden")

    def test_sharded_downstream_uses_native_norm_fusion(self):
        selected, reason = sharded_downstream_decision(
            requested=True,
            fused_moe_requires_full_hidden=True,
            backend_available=True,
            native_allreduce_rms_available=True,
        )
        self.assertTrue(selected)
        self.assertEqual(reason, "native_allreduce_rms")


if __name__ == "__main__":
    unittest.main()
