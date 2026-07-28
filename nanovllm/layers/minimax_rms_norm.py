import torch
import torch.distributed as dist
import triton
import triton.language as tl

from nanovllm import _router_gemm
from nanovllm.layers.compressed_collective import all_reduce
from nanovllm.layers.qk_workspace import peer_pointers, reset


_EPOCH = 0


def reset_qk_norm():
    global _EPOCH
    reset()
    _EPOCH = 0


@triton.jit
def _variance_kernel(
    qkv,
    variance,
    stride,
    q_size: tl.constexpr,
    kv_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    base = qkv + token * stride
    q_acc = 0.0
    for offset in range(0, q_size, BLOCK):
        index = offset + tl.arange(0, BLOCK)
        value = tl.load(base + index, mask=index < q_size, other=0.0)
        q_acc += tl.sum(value * value, axis=0)
    k_acc = 0.0
    for offset in range(0, kv_size, BLOCK):
        index = offset + tl.arange(0, BLOCK)
        value = tl.load(base + q_size + index, mask=index < kv_size, other=0.0)
        k_acc += tl.sum(value * value, axis=0)
    tl.store(variance + token * 2, q_acc / q_size)
    tl.store(variance + token * 2 + 1, k_acc / kv_size)


@triton.jit
def _apply_kernel(
    qkv,
    variance,
    q_weight,
    k_weight,
    q,
    k,
    stride,
    q_size: tl.constexpr,
    kv_size: tl.constexpr,
    world_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    base = qkv + token * stride
    q_inv = tl.rsqrt(tl.load(variance + token * 2) / world_size + eps)
    for offset in range(0, q_size, BLOCK):
        index = offset + tl.arange(0, BLOCK)
        mask = index < q_size
        value = tl.load(base + index, mask=mask)
        weight = tl.load(q_weight + index, mask=mask)
        tl.store(q + token * q_size + index, value * q_inv * weight, mask=mask)
    k_inv = tl.rsqrt(tl.load(variance + token * 2 + 1) / world_size + eps)
    for offset in range(0, kv_size, BLOCK):
        index = offset + tl.arange(0, BLOCK)
        mask = index < kv_size
        value = tl.load(base + q_size + index, mask=mask)
        weight = tl.load(k_weight + index, mask=mask)
        tl.store(k + token * kv_size + index, value * k_inv * weight, mask=mask)


def qk_rms_norm(qkv, q_weight, k_weight, q_size, kv_size, eps):
    global _EPOCH
    tokens = qkv.shape[0]
    if tokens <= 32:
        _EPOCH += 1
        q = torch.empty(tokens, q_size, dtype=qkv.dtype, device=qkv.device)
        k = torch.empty(tokens, kv_size, dtype=qkv.dtype, device=qkv.device)
        _router_gemm.qk_rms(
            qkv,
            q_weight,
            k_weight,
            q,
            k,
            peer_pointers(),
            dist.get_rank(),
            _EPOCH,
            eps,
        )
        return q, k
    variance = torch.empty(tokens, 2, dtype=torch.float32, device=qkv.device)
    _variance_kernel[(tokens,)](
        qkv,
        variance,
        qkv.stride(0),
        q_size,
        kv_size,
        1024,
    )
    all_reduce(variance.view(-1))
    q = torch.empty(tokens, q_size, dtype=qkv.dtype, device=qkv.device)
    k = torch.empty(tokens, kv_size, dtype=qkv.dtype, device=qkv.device)
    _apply_kernel[(tokens,)](
        qkv,
        variance,
        q_weight,
        k_weight,
        q,
        k,
        qkv.stride(0),
        q_size,
        kv_size,
        dist.get_world_size(),
        eps,
        1024,
    )
    return q, k
