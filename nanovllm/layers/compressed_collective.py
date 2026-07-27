import os

import torch
import torch.distributed as dist


BLOCK = 128
FP8_MAX = 448.0
DEFAULT_MIN_BYTES = 3 * 1024 * 1024
_STATS = {
    "compressed_calls": 0,
    "fallback_calls": 0,
    "flashinfer_calls": 0,
    "pynccl_calls": 0,
    "input_bytes": 0,
    "payload_bytes": 0,
}
_FI_WORKSPACE = None
_FI_WORKSPACE_KEY = None
_FI_DISABLED = False
_PYNCCL = None
_PYNCCL_GROUP = None
_PYNCCL_DISABLED = False
_VLLM_COMM = None
_VLLM_GROUP = None
_VLLM_DISABLED = False


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


def _flashinfer_workspace(x: torch.Tensor):
    """Lazily initialize FlashInfer's low-latency TP all-reduce workspace."""
    global _FI_WORKSPACE, _FI_WORKSPACE_KEY, _FI_DISABLED
    if _FI_DISABLED or os.getenv("NANOVLLM_ALL_REDUCE_BACKEND", "vllm") == "torch":
        return None
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        return None
    key = (dist.get_world_size(), x.shape[1], x.dtype, x.device)
    if _FI_WORKSPACE is not None and _FI_WORKSPACE_KEY == key:
        return _FI_WORKSPACE
    try:
        import flashinfer.comm as fi_comm
        from flashinfer.comm.mnnvl import TorchDistBackend

        backend = os.getenv("NANOVLLM_FLASHINFER_ALL_REDUCE_BACKEND", "trtllm")
        max_tokens = int(os.getenv("NANOVLLM_FLASHINFER_ALL_REDUCE_MAX_TOKENS", "2048"))
        group = dist.group.WORLD
        _FI_WORKSPACE = fi_comm.create_allreduce_fusion_workspace(
            backend=backend,
            world_size=dist.get_world_size(group),
            rank=dist.get_rank(group),
            max_token_num=max_tokens,
            hidden_dim=x.shape[1],
            dtype=x.dtype,
            comm_backend=TorchDistBackend(group=group),
            group=group,
        )
        _FI_WORKSPACE_KEY = key
        return _FI_WORKSPACE
    except Exception as exc:
        if os.getenv("NANOVLLM_ALL_REDUCE_DEBUG", "0") == "1":
            print(f"FlashInfer all-reduce workspace disabled: {exc!r}", flush=True)
        _FI_DISABLED = True
        return None


def _pynccl(x: torch.Tensor):
    """Use vLLM's direct NCCL communicator for small TP messages."""
    global _PYNCCL, _PYNCCL_GROUP, _PYNCCL_DISABLED
    if _PYNCCL_DISABLED or x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        return None
    if _PYNCCL is None:
        try:
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

            if _PYNCCL_GROUP is None:
                _PYNCCL_GROUP = dist.new_group(backend="gloo")
            _PYNCCL = PyNcclCommunicator(group=_PYNCCL_GROUP, device=x.device)
        except Exception:
            _PYNCCL_DISABLED = True
            return None
    try:
        if os.getenv("NANOVLLM_PYNCCL_INPLACE", "0") == "1":
            return _PYNCCL.all_reduce(x, out_tensor=x)
        # vLLM's CudaCommunicator uses PyNccl's out-of-place contract.
        return _PYNCCL.all_reduce(x)
    except Exception:
        _PYNCCL_DISABLED = True
        return None


def _vllm_communicator(x: torch.Tensor):
    """Build the same vLLM CudaCommunicator used by parallel_state."""
    global _VLLM_COMM, _VLLM_GROUP, _VLLM_DISABLED
    if _VLLM_DISABLED or not x.is_contiguous():
        return None
    if _VLLM_COMM is None:
        try:
            from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

            if _VLLM_GROUP is None:
                _VLLM_GROUP = dist.new_group(backend="gloo")
            _VLLM_COMM = CudaCommunicator(
                cpu_group=_VLLM_GROUP,
                device=x.device,
                device_group=dist.group.WORLD,
                unique_name="tp",
            )
        except Exception as exc:
            if os.getenv("NANOVLLM_ALL_REDUCE_DEBUG", "0") == "1":
                print(f"vLLM CUDA communicator disabled: {exc!r}", flush=True)
            _VLLM_DISABLED = True
            return None
    return _VLLM_COMM


def _vllm_cuda_communicator(x: torch.Tensor):
    global _VLLM_DISABLED
    communicator = _vllm_communicator(x)
    if communicator is None:
        return None
    try:
        return communicator.all_reduce(x)
    except Exception as exc:
        if os.getenv("NANOVLLM_ALL_REDUCE_DEBUG", "0") == "1":
            print(f"vLLM CUDA communicator failed: {exc!r}", flush=True)
        _VLLM_DISABLED = True
        return None


def all_reduce(x: torch.Tensor) -> torch.Tensor:
    selected, reason = _decision(x)
    if not selected:
        backend = os.getenv("NANOVLLM_ALL_REDUCE_BACKEND", "vllm")
        if backend == "vllm":
            out = _vllm_cuda_communicator(x)
            if out is not None:
                _record("pynccl_calls")
                return out
        if backend in ("auto", "pynccl"):
            out = _pynccl(x)
            if out is not None:
                _record("pynccl_calls")
                return out
        workspace = _flashinfer_workspace(x)
        if backend in ("auto", "flashinfer") and workspace is not None and x.numel() // x.shape[-1] <= int(os.getenv("NANOVLLM_FLASHINFER_ALL_REDUCE_MAX_TOKENS", "2048")):
            try:
                import flashinfer.comm as fi_comm
                out = fi_comm.allreduce_fusion(
                    input=x,
                    workspace=workspace,
                    pattern=fi_comm.AllReduceFusionPattern.kAllReduce,
                    launch_with_pdl=True,
                    trigger_completion_at_end=True,
                )
                _record("flashinfer_calls")
                return out
            except Exception:
                pass
        _record("fallback_calls", int(reason != "disabled"))
        dist.all_reduce(x)
        return x

    world_size = dist.get_world_size()
    communicator = _vllm_communicator(x)
    pynccl = getattr(communicator, "pynccl_comm", None)
    if pynccl is None or getattr(pynccl, "disabled", False):
        _record("fallback_calls")
        dist.all_reduce(x)
        return x
    flat = x.contiguous().view(-1, BLOCK)
    scale = flat.float().abs().amax(dim=1, keepdim=True)
    scale = pynccl.all_reduce(scale, op=dist.ReduceOp.MAX)
    scale.clamp_min_(1e-12).div_(FP8_MAX / world_size)
    quantized = (flat / scale).clamp_(-FP8_MAX / world_size, FP8_MAX / world_size).to(torch.float8_e4m3fn)
    try:
        reduced = pynccl.all_reduce(quantized)
    except RuntimeError:
        _record("fallback_calls")
        dist.all_reduce(x)
        return x
    x.copy_((reduced.float() * scale).view_as(x).to(x.dtype))
    _record("compressed_calls")
    _record("input_bytes", x.numel() * x.element_size())
    _record("payload_bytes", quantized.numel() + scale.numel() * scale.element_size())
    return x


def all_gather(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Match vLLM's TP logits all-gather (PyNccl when available)."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return x
    if dim < 0:
        dim += x.ndim
    if not 0 <= dim < x.ndim:
        raise ValueError(f"invalid all-gather dim {dim} for {tuple(x.shape)}")
    communicator = _vllm_communicator(x) if os.getenv("NANOVLLM_LOGITS_BACKEND", "vllm") == "vllm" else None
    pynccl = getattr(communicator, "pynccl_comm", None)
    if pynccl is not None and not getattr(pynccl, "disabled", False):
        input_size = tuple(x.shape)
        output_size = (input_size[0] * dist.get_world_size(),) + input_size[1:]
        output = torch.empty(output_size, dtype=x.dtype, device=x.device)
        pynccl.all_gather(output, x.contiguous())
        return output.reshape((dist.get_world_size(),) + input_size).movedim(0, dim).reshape(
            input_size[:dim] + (dist.get_world_size() * input_size[dim],) + input_size[dim + 1:]
        )
    shards = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(shards, x.contiguous())
    return torch.cat(shards, dim=dim)
