import unittest

import torch
from torch import nn

from nanovllm.utils.loader import initialize_dummy_weights


class DummyLoaderTest(unittest.TestCase):

    def test_synthetic_weights_are_deterministic_and_nonzero(self):
        first = nn.Linear(8, 4, bias=False)
        second = nn.Linear(8, 4, bias=False)
        initialize_dummy_weights(first, seed=1234)
        initialize_dummy_weights(second, seed=1234)
        torch.testing.assert_close(first.weight, second.weight, rtol=0, atol=0)
        self.assertGreater(torch.count_nonzero(first.weight).item(), 0)

    def test_one_dimensional_norm_weights_are_one(self):
        module = nn.Sequential(
            nn.LayerNorm(8, elementwise_affine=True, bias=False)
        )
        initialize_dummy_weights(module, seed=1234)
        torch.testing.assert_close(
            module[0].weight,
            torch.ones_like(module[0].weight),
            rtol=0,
            atol=0,
        )
