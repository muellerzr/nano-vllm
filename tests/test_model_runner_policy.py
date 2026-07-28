import unittest
from types import SimpleNamespace

from nanovllm.engine.model_runner import (
    _has_real_block_tables,
    _warmup_sequence_lengths,
)


class ModelRunnerPolicyTest(unittest.TestCase):

    def test_warmup_preserves_token_count_without_long_ragged_queries(self):
        lengths = _warmup_sequence_lengths(2048)
        self.assertEqual(lengths, [512, 512, 512, 512])
        self.assertEqual(sum(lengths), 2048)

    def test_warmup_handles_a_remainder(self):
        self.assertEqual(_warmup_sequence_lengths(1025), [512, 512, 1])

    def test_real_requests_use_paged_prefill_from_the_first_chunk(self):
        self.assertTrue(
            _has_real_block_tables([SimpleNamespace(block_table=[4, 5])])
        )
        self.assertFalse(
            _has_real_block_tables([SimpleNamespace(block_table=[])])
        )
