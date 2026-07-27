import os

import torch
import torch.distributed as dist


BLOCK = 128
FP8_MAX = 448.0
DEFAULT_MIN_BYTES = 3 * 1024 * 1024
_STATS = {"compressed_calls": 0, "fallback_calls": 0, "input_bytes": 0, "payload_bytes": 0}


def _record(key: str, value: int = 1) -> None:
    if os.getenv("NANOVLLM_FP8_ALL_REDUCE_STATS", "0") == "1":
        _STATS[key] += value


def _decision(x: torch.Tensor) -> tuple[bool, str]:
    if os.getenv("NANOVLLM_FP8_ALL_REDUCE", "0") != "1":
        return False, "disabled"
    if x.dtype != torch.bfloat16:
        return False, "dtype"
    if not x.is_cuda:
        return False, "device"
    if x.numel() * x.element_size() < int(os.getenv("NANOVLLM_FP8_ALL_REDUCE_MIN_BYTES", str(DEFAULT_MIN_BYTES))):
        return False, "threshold"
    if not x.is_contiguous() or x.numel() % BLOCK:
        return False, "layout"
    if not dist.is_initialized() or dist.get_backend() != "nccl":
        return False, "backend"
    if torch.cuda.get_device_capability(x.device)[0] < 9:
        return False, "capability"
    return True, "eligible"


def reset_stats() -> None:
    for key in _STATS:
        _STATS[key] = 0


def stats() -> dict[str, int | str]:
    return {**_STATS, "min_bytes": int(os.getenv("NANOVLLM_FP8_ALL_REDUCE_MIN_BYTES", str(DEFAULT_MIN_BYTES))), "block_size": BLOCK}


def all_reduce(x: torch.Tensor) -> torch.Tensor:
    selected, reason = _decision(x)
    if not selected:
        _record("fallback_calls", int(reason != "disabled"))
        dist.all_reduce(x)
        return x

    world_size = dist.get_world_size()
    flat = x.contiguous().view(-1, BLOCK)
    scale = flat.float().abs().amax(dim=1, keepdim=True)
    dist.all_reduce(scale, op=dist.ReduceOp.MAX)
    scale.clamp_min_(1e-12).div_(FP8_MAX / world_size)
    quantized = (flat / scale).clamp_(-FP8_MAX / world_size, FP8_MAX / world_size).to(torch.float8_e4m3fn)
    try:
        dist.all_reduce(quantized)
    except RuntimeError:
        _record("fallback_calls")
        dist.all_reduce(x)
        return x
    x.copy_((quantized.float() * scale).view_as(x).to(x.dtype))
    _record("compressed_calls")
    _record("input_bytes", x.numel() * x.element_size())
    _record("payload_bytes", quantized.numel() + scale.numel() * scale.element_size())
    return x
