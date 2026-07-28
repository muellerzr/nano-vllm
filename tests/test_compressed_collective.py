import unittest
from unittest.mock import Mock, patch

import torch

from nanovllm.layers import compressed_collective as collective


class CompressedCollectiveTest(unittest.TestCase):

    def tearDown(self):
        collective.configure()
        collective._COMM = None
        collective._GROUP = None

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

    def test_shutdown_releases_communicator_and_group(self):
        communicator = Mock()
        group = object()
        collective._COMM = communicator
        collective._GROUP = group
        with patch.object(collective.dist, "destroy_process_group") as destroy:
            collective.shutdown()
        communicator.close.assert_called_once_with()
        destroy.assert_called_once_with(group)
        self.assertIsNone(collective._COMM)
        self.assertIsNone(collective._GROUP)
