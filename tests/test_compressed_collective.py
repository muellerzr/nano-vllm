import torch
import torch.distributed as dist

from nanovllm.layers.compressed_collective import BLOCK, DEFAULT_MIN_BYTES, _decision, all_reduce, stats


def test_disabled_is_the_default(monkeypatch):
    monkeypatch.delenv("NANOVLLM_FP8_ALL_REDUCE", raising=False)
    selected, reason = _decision(torch.zeros(BLOCK, dtype=torch.bfloat16))
    assert not selected
    assert reason == "disabled"


def test_cpu_and_small_payloads_fall_back(monkeypatch):
    monkeypatch.setenv("NANOVLLM_FP8_ALL_REDUCE", "1")
    selected, reason = _decision(torch.zeros(BLOCK, dtype=torch.bfloat16))
    assert not selected
    assert reason == "device"
    assert DEFAULT_MIN_BYTES == 3 * 1024 * 1024


def test_disabled_path_calls_stock_collective(monkeypatch):
    called = []
    monkeypatch.delenv("NANOVLLM_FP8_ALL_REDUCE", raising=False)
    monkeypatch.setattr(dist, "all_reduce", lambda tensor: called.append(tensor))
    tensor = torch.zeros(BLOCK, dtype=torch.bfloat16)
    assert all_reduce(tensor) is tensor
    assert called == [tensor]


def test_stats_shape_is_stable(monkeypatch):
    monkeypatch.setenv("NANOVLLM_FP8_ALL_REDUCE_MIN_BYTES", str(DEFAULT_MIN_BYTES))
    result = stats()
    assert result["min_bytes"] == DEFAULT_MIN_BYTES
    assert result["block_size"] == BLOCK
