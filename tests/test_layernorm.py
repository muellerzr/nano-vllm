import unittest

import torch

from nanovllm.layers.layernorm import RMSNorm


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class RMSNormTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.x = torch.randn(7, 3072, dtype=torch.bfloat16, device="cuda")
        self.residual = torch.randn_like(self.x)
        self.weight = torch.randn(3072, dtype=torch.bfloat16, device="cuda")

    @staticmethod
    def reference(x, weight, eps):
        value = x.float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)).to(x.dtype) * weight

    def make_norm(self):
        norm = RMSNorm(self.x.shape[-1]).cuda().bfloat16()
        norm.weight.data.copy_(self.weight)
        return norm

    def test_rms_norm_matches_reference(self):
        norm = self.make_norm()
        actual = norm(self.x.clone())
        expected = self.reference(self.x, self.weight, norm.eps)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_fused_add_rms_norm_matches_reference(self):
        norm = self.make_norm()
        x = self.x.clone()
        residual = self.residual.clone()
        actual, actual_residual = norm(x, residual)
        expected_residual = (self.x.float() + self.residual.float()).to(torch.bfloat16)
        expected = self.reference(expected_residual, self.weight, norm.eps)
        self.assertIs(actual, x)
        self.assertIs(actual_residual, residual)
        torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

if __name__ == "__main__":
    unittest.main()
