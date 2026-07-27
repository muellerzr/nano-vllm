"""MiniMax TP Q/K RMSNorm path copied from vLLM's implementation.

The norm is over the full tensor-parallel Q/K vector.  Each rank computes its
local mean of squares, then the means are reduced before the final scale.  If
the pinned vLLM extension and a Lamport workspace are available, the fused
vLLM kernel is used; otherwise the same two Triton kernels surround a TP
all-reduce.
"""

import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

try:
    # Registers the pinned vLLM custom op when its wheel is present.  The
    # implementation below remains self-contained when it is not.
    import vllm.model_executor.layers.minimax_rms_norm  # noqa: F401
except ImportError:
    pass


MAX_TOKENS = 2048
_FUSED = getattr(torch.ops._C, "minimax_allreduce_rms_qk", None)
_WORKSPACE = None
_WORKSPACE_GROUP = None
_WORKSPACE_TRIED = False


@triton.jit
def _variance_kernel(qkv_ptr, var_ptr, row_stride, q_size: tl.constexpr,
                     kv_size: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    base = qkv_ptr + token * row_stride
    q_acc = 0.0
    for off in range(0, q_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < q_size
        x = tl.load(base + idx, mask=mask, other=0.0).to(tl.float32)
        q_acc += tl.sum(x * x, axis=0)
    k_acc = 0.0
    for off in range(0, kv_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < kv_size
        x = tl.load(base + q_size + idx, mask=mask, other=0.0).to(tl.float32)
        k_acc += tl.sum(x * x, axis=0)
    tl.store(var_ptr + token * 2, q_acc / q_size)
    tl.store(var_ptr + token * 2 + 1, k_acc / kv_size)


@triton.jit
def _apply_kernel(qkv_ptr, var_ptr, q_weight_ptr, k_weight_ptr, q_out_ptr,
                  k_out_ptr, row_stride, q_size: tl.constexpr,
                  kv_size: tl.constexpr, tp_world: tl.constexpr,
                  eps: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    base = qkv_ptr + token * row_stride
    q_inv = tl.rsqrt(tl.load(var_ptr + token * 2) / tp_world + eps)
    for off in range(0, q_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < q_size
        x = tl.load(base + idx, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(q_weight_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        tl.store(q_out_ptr + token * q_size + idx, (x * q_inv * w).to(q_out_ptr.dtype.element_ty), mask=mask)
    k_inv = tl.rsqrt(tl.load(var_ptr + token * 2 + 1) / tp_world + eps)
    for off in range(0, kv_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < kv_size
        x = tl.load(base + q_size + idx, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(k_weight_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        tl.store(k_out_ptr + token * kv_size + idx, (x * k_inv * w).to(k_out_ptr.dtype.element_ty), mask=mask)


def _workspace() -> torch.Tensor | None:
    global _WORKSPACE, _WORKSPACE_GROUP, _WORKSPACE_TRIED
    if _WORKSPACE_TRIED or _FUSED is None or dist.get_world_size() <= 1:
        return _WORKSPACE
    _WORKSPACE_TRIED = True
    if os.getenv("NANOVLLM_QK_NORM_BACKEND", "vllm") in ("fallback", "triton"):
        return None
    try:
        from vllm.model_executor.layers.minimax_rms_norm.lamport_workspace import (
            get_allreduce_workspace,
        )

        _WORKSPACE_GROUP = dist.new_group(backend="gloo")
        _WORKSPACE = get_allreduce_workspace(
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            max_tokens=MAX_TOKENS,
            process_group=_WORKSPACE_GROUP,
        )
    except Exception:
        _WORKSPACE = None
    return _WORKSPACE


def qk_rms_norm(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_size: int,
    kv_size: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run vLLM's MiniMax all-reduce Q/K RMSNorm contract."""
    if qkv.ndim != 2:
        raise ValueError("MiniMax Q/K RMSNorm expects a 2-D QKV tensor")
    world = dist.get_world_size()
    workspace = _workspace()
    if workspace is not None and qkv.shape[0] <= MAX_TOKENS:
        return _FUSED(
            qkv, q_weight, k_weight, workspace, q_size, kv_size,
            dist.get_rank(), world, eps,
        )

    if world == 1:
        q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
        qf, kf = q.float(), k.float()
        return (
            (qf * torch.rsqrt(qf.square().mean(-1, keepdim=True) + eps)).to(q.dtype) * q_weight,
            (kf * torch.rsqrt(kf.square().mean(-1, keepdim=True) + eps)).to(k.dtype) * k_weight,
        )

    tokens = qkv.shape[0]
    variance = torch.empty(tokens, 2, dtype=torch.float32, device=qkv.device)
    _variance_kernel[(tokens,)](
        qkv, variance, qkv.stride(0), q_size=q_size, kv_size=kv_size, BLOCK=1024,
    )
    dist.all_reduce(variance.view(-1))
    q = torch.empty(tokens, q_size, dtype=qkv.dtype, device=qkv.device)
    k = torch.empty(tokens, kv_size, dtype=qkv.dtype, device=qkv.device)
    _apply_kernel[(tokens,)](
        qkv, variance, q_weight, k_weight, q, k, qkv.stride(0),
        q_size=q_size, kv_size=kv_size, tp_world=world, eps=eps, BLOCK=1024,
    )
    return q, k
