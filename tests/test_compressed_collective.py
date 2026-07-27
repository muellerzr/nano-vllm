import os
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

from nanovllm.layers.compressed_collective import BLOCK, DEFAULT_MIN_BYTES, _decision, all_reduce, stats


class CompressedCollectiveTest(unittest.TestCase):

    def test_disabled_is_the_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NANOVLLM_FP8_ALL_REDUCE", None)
            selected, reason = _decision(torch.zeros(BLOCK, dtype=torch.bfloat16))
        self.assertFalse(selected)
        self.assertEqual(reason, "disabled")


    def test_cpu_and_small_payloads_fall_back(self):
        with patch.dict(os.environ, {"NANOVLLM_FP8_ALL_REDUCE": "1"}):
            selected, reason = _decision(torch.zeros(BLOCK, dtype=torch.bfloat16))
        self.assertFalse(selected)
        self.assertEqual(reason, "device")
        self.assertEqual(DEFAULT_MIN_BYTES, 3 * 1024 * 1024)


    def test_disabled_path_calls_stock_collective(self):
        called = []
        with patch.dict(os.environ, {}, clear=False), patch.object(dist, "all_reduce", lambda tensor: called.append(tensor)):
            os.environ.pop("NANOVLLM_FP8_ALL_REDUCE", None)
            tensor = torch.zeros(BLOCK, dtype=torch.bfloat16)
            self.assertIs(all_reduce(tensor), tensor)
        self.assertEqual(called, [tensor])


    def test_stats_shape_is_stable(self):
        with patch.dict(os.environ, {"NANOVLLM_FP8_ALL_REDUCE_MIN_BYTES": str(DEFAULT_MIN_BYTES)}):
            result = stats()
        self.assertEqual(result["min_bytes"], DEFAULT_MIN_BYTES)
        self.assertEqual(result["block_size"], BLOCK)
