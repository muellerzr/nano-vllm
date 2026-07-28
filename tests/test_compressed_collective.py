import unittest
from unittest.mock import patch

import torch

from nanovllm.layers import compressed_collective as collective


class CompressedCollectiveTest(unittest.TestCase):

    def tearDown(self):
        collective.configure()

    def test_disabled_is_default(self):
        collective.configure()
        selected, reason = collective._decision(
            torch.zeros(collective.BLOCK, dtype=torch.bfloat16)
        )
        self.assertFalse(selected)
        self.assertEqual(reason, "disabled")

    def test_threshold_is_configurable(self):
        threshold = collective.DEFAULT_MIN_BYTES + 256
        collective.configure(min_bytes=threshold)
        self.assertEqual(collective.stats()["min_bytes"], threshold)

    def test_disabled_reduce_is_in_place(self):
        class Communicator:
            def all_reduce(self, tensor, out=None, op=None):
                self.out = out
                tensor.add_(3)
                return tensor

        communicator = Communicator()
        with patch.object(collective, "_communicator", return_value=communicator):
            collective.configure()
            source = torch.ones(8, dtype=torch.bfloat16)
            result = collective.all_reduce(source)
        self.assertIs(result, source)
        self.assertIsNone(communicator.out)
        torch.testing.assert_close(
            result,
            torch.full_like(result, 4),
            rtol=0,
            atol=0,
        )
