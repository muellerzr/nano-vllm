import unittest

from vllm_overlay.policy import (
    compression_decision,
    payload_bytes,
)


class CompressionPolicyTests(unittest.TestCase):
    def test_disabled_path_is_stock(self):
        self.assertEqual(
            compression_decision(
                enabled=False,
                dtype="torch.bfloat16",
                is_cuda=True,
                nbytes=8 << 20,
                numel=8 << 20,
                is_contiguous=True,
                capability_major=12,
                backend_available=True,
            ),
            (False, "disabled"),
        )

    def test_threshold_is_inclusive(self):
        kwargs = dict(
            enabled=True,
            dtype="torch.bfloat16",
            is_cuda=True,
            nbytes=3 << 20,
            numel=3 << 20,
            is_contiguous=True,
            capability_major=12,
            backend_available=True,
        )
        self.assertEqual(compression_decision(**kwargs), (True, "eligible"))
        kwargs["nbytes"] -= 1
        self.assertEqual(compression_decision(**kwargs), (False, "threshold"))

    def test_unsupported_inputs_fallback_with_reason(self):
        base = dict(
            enabled=True,
            dtype="torch.bfloat16",
            is_cuda=True,
            nbytes=8 << 20,
            numel=8 << 20,
            is_contiguous=True,
            capability_major=12,
            backend_available=True,
        )
        for field, value, reason in [
            ("dtype", "torch.float16", "dtype"),
            ("is_cuda", False, "device"),
            ("is_contiguous", False, "layout"),
            ("capability_major", 8, "capability"),
            ("backend_available", False, "backend"),
        ]:
            case = dict(base)
            case[field] = value
            self.assertEqual(compression_decision(**case), (False, reason))

    def test_payload_accounts_for_fp8_values_and_fp32_scales(self):
        self.assertEqual(payload_bytes(128), 128 + 4)
        self.assertEqual(payload_bytes(256), 256 + 8)
        self.assertEqual(payload_bytes(129), 129 + 8)


if __name__ == "__main__":
    unittest.main()
