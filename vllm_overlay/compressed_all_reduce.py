import os
from collections import Counter

import torch
from torch.distributed import ReduceOp

from .policy import compression_decision, payload_bytes


_STATS = Counter()
_ENABLED = os.getenv("VLLM_FP8_ALL_REDUCE", "0") == "1"
_MIN_BYTES = int(
    os.getenv("VLLM_FP8_ALL_REDUCE_MIN_BYTES", str(3 * 1024 * 1024))
)
_STATS_ENABLED = os.getenv("VLLM_FP8_ALL_REDUCE_STATS", "0") == "1"
# Imported by the patched vLLM entry point so sub-threshold decode reductions
# can bypass the helper call entirely.
FP8_ALL_REDUCE_ENABLED = _ENABLED
FP8_ALL_REDUCE_MIN_BYTES = _MIN_BYTES


def _stats_enabled() -> bool:
    return _STATS_ENABLED


def reset_stats() -> None:
    _STATS.clear()


def get_stats() -> dict[str, int | str]:
    return {
        **_STATS,
        "enabled": int(_ENABLED),
        "min_bytes": _MIN_BYTES,
        "block_size": 128,
        "format": "float8_e4m3fn",
    }


def _fallback(reason: str) -> None:
    if not _stats_enabled():
        return
    _STATS["fallback_calls"] += 1
    _STATS[f"fallback_{reason}"] += 1


def maybe_compressed_all_reduce(
    tensor: torch.Tensor,
    group,
) -> torch.Tensor | None:
    enabled = _ENABLED
    if not enabled:
        return None

    nbytes = tensor.numel() * tensor.element_size()
    threshold = _MIN_BYTES
    # Decode reductions are far below the compression crossover on this
    # topology.  Keep the feature-off behavior genuinely cheap: do not query
    # communicator/capability state or update diagnostic counters for a
    # payload that cannot be compressed.  Diagnostics remain available via
    # VLLM_FP8_ALL_REDUCE_STATS=1.
    if nbytes < threshold:
        if _stats_enabled():
            _STATS["considered_calls"] += 1
            _fallback("threshold")
        return None
    communicator = getattr(group.device_communicator, "pynccl_comm", None)
    backend_available = communicator is not None and not communicator.disabled
    selected, reason = compression_decision(
        enabled=enabled,
        dtype=str(tensor.dtype),
        is_cuda=tensor.is_cuda,
        nbytes=nbytes,
        numel=tensor.numel(),
        is_contiguous=tensor.is_contiguous(),
        capability_major=(
            torch.cuda.get_device_capability(tensor.device)[0]
            if tensor.is_cuda
            else 0
        ),
        backend_available=backend_available,
        min_bytes=threshold,
    )
    if not selected:
        if _stats_enabled():
            _STATS["considered_calls"] += 1
        _fallback(reason)
        return None

    if _stats_enabled():
        _STATS["considered_calls"] += 1

    blocks = tensor.view(-1, 128)
    scale = blocks.float().abs().amax(dim=1, keepdim=True)
    scale = communicator.all_reduce(scale, op=ReduceOp.MAX)
    scale = scale.clamp_min_(1e-12).div_(448.0 / group.world_size)
    quantized = (blocks / scale).clamp_(
        -448.0 / group.world_size,
        448.0 / group.world_size,
    ).to(torch.float8_e4m3fn)
    reduced = communicator.all_reduce(quantized)
    output = (reduced.float() * scale).to(tensor.dtype).view_as(tensor)

    if _stats_enabled():
        _STATS["compressed_calls"] += 1
        _STATS["input_bytes"] += nbytes
        _STATS["payload_bytes"] += payload_bytes(quantized.numel())
    return output
