import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nanovllm.models import minimax_m2


class Linear(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.zeros(1, dtype=torch.long)
        self.weight_scale_inv = torch.zeros(1, dtype=torch.int32)
        self.weight_global_scale = torch.ones(1)
        self.input_global_scale = torch.ones(1)

    def forward(self, x):
        return x


class MiniMaxMoeTest(unittest.TestCase):

    def make_block(self):
        config = SimpleNamespace(num_experts_per_tok=8, num_local_experts=256, hidden_size=3072, intermediate_size=1536)
        with (
            patch.object(minimax_m2, "RouterLinear", return_value=Linear()),
            patch.object(minimax_m2, "NVFP4GroupedLinear", return_value=Linear()),
            patch.object(minimax_m2.dist, "get_world_size", return_value=4),
            patch.object(minimax_m2.dist, "get_rank", return_value=0),
        ):
            return minimax_m2.MiniMaxM2SparseMoeBlock(config)

    def test_workspace_does_not_replace_module_buffers(self):
        block = self.make_block()
        self.assertIsInstance(block._buffers, dict)
        self.assertIsNone(block.workspace)

    def test_flashinfer_result_is_unwrapped(self):
        block = self.make_block()
        block._flashinfer_moe = lambda **kwargs: [kwargs["output"]]
        hidden_states = torch.zeros(2, 3072)
        with (
            patch.object(minimax_m2, "topk_sigmoid"),
            patch.object(minimax_m2, "all_reduce", side_effect=lambda output: output),
        ):
            output = block(hidden_states)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, hidden_states.shape)


if __name__ == "__main__":
    unittest.main()
