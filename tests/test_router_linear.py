import unittest

import torch
import torch.nn.functional as F

from nanovllm.layers.linear import RouterLinear


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class RouterLinearTest(unittest.TestCase):

    def test_specialized_router_matches_fp32_linear(self):
        torch.manual_seed(1234)
        layer = RouterLinear(3072, 256).cuda()
        layer.weight.data.normal_()
        for tokens in (1, 8, 17, 32):
            x = torch.randn(tokens, 3072, dtype=torch.bfloat16, device="cuda")
            expected = F.linear(x.float(), layer.weight)
            actual = layer(x)
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-3)

    def test_large_router_matches_fp32_linear(self):
        torch.manual_seed(1234)
        layer = RouterLinear(3072, 256).cuda()
        layer.weight.data.normal_()
        x = torch.randn(33, 3072, dtype=torch.bfloat16, device="cuda")
        torch.testing.assert_close(layer(x), F.linear(x.float(), layer.weight))


if __name__ == "__main__":
    unittest.main()
