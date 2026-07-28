import torch
import torch.distributed as dist

from nanovllm.layers.pynccl import NcclCommunicator


BLOCK = 128
FP8_MAX = 448.0
DEFAULT_MIN_BYTES = 3 * 1024 * 1024
_COMM = None
_GROUP = None
_ENABLED = False
_MIN_BYTES = DEFAULT_MIN_BYTES
_STATS_ENABLED = False
_STATS = {
    "compressed_calls": 0,
    "fallback_calls": 0,
    "pynccl_calls": 0,
    "input_bytes": 0,
    "payload_bytes": 0,
}


def configure(enabled=False, min_bytes=DEFAULT_MIN_BYTES, record_stats=False):
    global _ENABLED, _MIN_BYTES, _STATS_ENABLED
    _ENABLED = enabled
    _MIN_BYTES = min_bytes
    _STATS_ENABLED = record_stats


def _record(key, value=1):
    if _STATS_ENABLED:
        _STATS[key] += value


def _communicator(x):
    global _COMM, _GROUP
    if _COMM is None:
        _GROUP = dist.new_group(backend="gloo")
        _COMM = NcclCommunicator(_GROUP, x.device)
    return _COMM


def _decision(x):
    if not _ENABLED:
        return False, "disabled"
    if x.dtype != torch.bfloat16:
        return False, "dtype"
    if x.numel() * x.element_size() < _MIN_BYTES:
        return False, "threshold"
    if not x.is_contiguous() or x.numel() % BLOCK:
        return False, "layout"
    return True, "eligible"


def reset_stats():
    for key in _STATS:
        _STATS[key] = 0


def shutdown():
    global _COMM, _GROUP
    if _COMM is None:
        return
    _COMM.close()
    dist.destroy_process_group(_GROUP)
    _COMM = None
    _GROUP = None


def stats():
    return {**_STATS, "min_bytes": _MIN_BYTES, "block_size": BLOCK}


def all_reduce(x):
    selected, reason = _decision(x)
    comm = _communicator(x)
    if not selected:
        comm.all_reduce(x)
        _record("pynccl_calls")
        _record("fallback_calls", int(reason != "disabled"))
        return x

    flat = x.view(-1, BLOCK)
    scale = flat.float().abs().amax(dim=1, keepdim=True)
    comm.all_reduce(scale, op=dist.ReduceOp.MAX)
    scale.clamp_min_(1e-12).div_(FP8_MAX / dist.get_world_size())
    quantized = (
        (flat / scale)
        .clamp_(-FP8_MAX / dist.get_world_size(), FP8_MAX / dist.get_world_size())
        .to(torch.float8_e4m3fn)
    )
    reduced = torch.empty_like(quantized)
    comm.all_reduce(quantized, reduced)
    x.copy_((reduced.float() * scale).view_as(x))
    _record("compressed_calls")
    _record("input_bytes", x.numel() * x.element_size())
    _record(
        "payload_bytes",
        quantized.numel() + scale.numel() * scale.element_size(),
    )
    return x


def all_gather(x, dim=-1):
    if dim < 0:
        dim += x.ndim
    input_shape = tuple(x.shape)
    output = torch.empty(
        (input_shape[0] * dist.get_world_size(),) + input_shape[1:],
        dtype=x.dtype,
        device=x.device,
    )
    _communicator(x).all_gather(output, x.contiguous())
    return (
        output.reshape((dist.get_world_size(),) + input_shape)
        .movedim(0, dim)
        .reshape(
            input_shape[:dim]
            + (dist.get_world_size() * input_shape[dim],)
            + input_shape[dim + 1 :]
        )
    )
