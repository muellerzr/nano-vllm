import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl


@triton.jit
def quant_fp8_kernel(x, y, scales, k: tl.constexpr, block_size: tl.constexpr):
    row = tl.program_id(0)
    group = tl.program_id(1)
    offsets = group * block_size + tl.arange(0, block_size)
    values = tl.load(x + row * k + offsets).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values), axis=0), 1e-12) / 448.0
    tl.store(y + row * k + offsets, values / scale)
    tl.store(scales + row * (k // block_size) + group, scale)


def quant_fp8(x: torch.Tensor, block_size: int = 128):
    assert x.is_cuda and x.is_contiguous() and x.ndim == 2
    assert x.shape[1] % block_size == 0
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        x.shape[0],
        x.shape[1] // block_size,
        dtype=torch.float32,
        device=x.device,
    )
    quant_fp8_kernel[(x.shape[0], x.shape[1] // block_size)](
        x,
        y,
        scales,
        x.shape[1],
        block_size,
    )
    return y, scales


@triton.jit
def fp8_scales_kernel(
    x,
    scales,
    k: tl.constexpr,
    block_size: tl.constexpr,
    max_value: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    offsets = group * block_size + tl.arange(0, block_size)
    values = tl.load(x + row * k + offsets).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values), axis=0), 1e-12) / max_value
    tl.store(scales + row * (k // block_size) + group, scale)


@triton.jit
def quant_fp8_with_scales_kernel(
    x,
    y,
    scales,
    k: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    offsets = group * block_size + tl.arange(0, block_size)
    scale = tl.load(scales + row * (k // block_size) + group)
    values = tl.load(x + row * k + offsets).to(tl.float32)
    tl.store(y + row * k + offsets, values / scale)


@triton.jit
def dequant_fp8_kernel(
    x,
    y,
    scales,
    k: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    offsets = group * block_size + tl.arange(0, block_size)
    scale = tl.load(scales + row * (k // block_size) + group)
    values = tl.load(x + row * k + offsets).to(tl.float32)
    tl.store(y + row * k + offsets, values * scale)


def fp8_all_reduce(x: torch.Tensor, block_size: int = 128):
    shape = x.shape
    x = x.reshape(-1, shape[-1]).contiguous()
    rows, k = x.shape
    groups = k // block_size
    scales = torch.empty(rows, groups, dtype=torch.float32, device=x.device)
    fp8_scales_kernel[(rows, groups)](
        x,
        scales,
        k,
        block_size,
        448.0 / dist.get_world_size(),
    )
    dist.all_reduce(scales, op=dist.ReduceOp.MAX)
    quantized = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    quant_fp8_with_scales_kernel[(rows, groups)](
        x,
        quantized,
        scales,
        k,
        block_size,
    )
    dist.all_reduce(quantized)
    output = torch.empty_like(x)
    dequant_fp8_kernel[(rows, groups)](
        quantized,
        output,
        scales,
        k,
        block_size,
    )
    return output.reshape(shape)


def all_reduce(x: torch.Tensor):
    threshold = int(
        os.getenv(
            "NANOVLLM_FP8_ALL_REDUCE_MIN_BYTES",
            str(3 * 1024 * 1024),
        )
    )
    if os.getenv("NANOVLLM_FP8_ALL_REDUCE", "1") == "1" and x.numel() * x.element_size() >= threshold:
        return fp8_all_reduce(x)
    dist.all_reduce(x)
    return x


@triton.jit
def block_scaled_mm_kernel(
    x,
    weight,
    x_scales,
    weight_scales,
    output,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * block_m + tl.arange(0, block_m)
    offsets_n = pid_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for block in range(0, tl.cdiv(k, block_k)):
        x_values = tl.load(
            x + offsets_m[:, None] * k + block * block_k + offsets_k[None, :],
            mask=offsets_m[:, None] < m,
            other=0.0,
        )
        weight_values = tl.load(
            weight + offsets_n[:, None] * k + block * block_k + offsets_k[None, :],
            mask=offsets_n[:, None] < n,
            other=0.0,
        )
        x_scale = tl.load(
            x_scales + offsets_m * (k // block_k) + block,
            mask=offsets_m < m,
            other=0.0,
        )
        weight_scale = tl.load(
            weight_scales + (offsets_n // block_n) * (k // block_k) + block,
            mask=offsets_n < n,
            other=0.0,
        )
        accumulator += tl.dot(x_values, tl.trans(weight_values)) * x_scale[:, None] * weight_scale[None, :]
    tl.store(
        output + offsets_m[:, None] * n + offsets_n[None, :],
        accumulator,
        mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n),
    )


def block_scaled_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scales: torch.Tensor,
    weight_scales: torch.Tensor,
):
    assert x.is_cuda and x.is_contiguous() and weight.is_contiguous()
    m, k = x.shape
    n = weight.shape[0]
    block_m = 16 if m < 64 else 64
    block_n = block_k = 128
    output = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    block_scaled_mm_kernel[(triton.cdiv(m, block_m), triton.cdiv(n, block_n))](
        x,
        weight,
        x_scales,
        weight_scales,
        output,
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
    )
    return output
