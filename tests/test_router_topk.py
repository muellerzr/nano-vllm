import unittest

import torch

from nanovllm.layers.router import topk_sigmoid


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class RouterTopKTest(unittest.TestCase):

    def test_topk_sigmoid_matches_torch(self):
        torch.manual_seed(1234)
        logits = torch.randn(17, 256, dtype=torch.float32, device="cuda")
        scores = torch.sigmoid(logits)
        expected_weights, expected_indices = torch.topk(scores, 8, dim=-1)
        expected_weights /= expected_weights.sum(-1, keepdim=True)
        weights = torch.empty_like(expected_weights)
        indices = torch.empty_like(expected_indices, dtype=torch.int32)
        topk_sigmoid(logits, weights, indices)
        torch.testing.assert_close(weights, expected_weights, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(indices, expected_indices.int(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
