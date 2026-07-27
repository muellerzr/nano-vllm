import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.nn.functional import ScalingType, SwizzleType


@triton.jit
def block_scales_kernel(
    x,
    scales,
    k: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_size + tl.arange(0, block_size)
    values = tl.load(x + row * k + offsets).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values), axis=0), 1e-12) / 6.0
    tl.store(scales + row * (k // block_size) + block, scale)


@triton.jit
def fp4_code(value):
    value = tl.abs(value)
    code = tl.where(value < 0.25, 0, 1)
    code = tl.where(value < 0.75, code, 2)
    code = tl.where(value < 1.25, code, 3)
    code = tl.where(value < 1.75, code, 4)
    code = tl.where(value < 2.5, code, 5)
    code = tl.where(value < 3.5, code, 6)
    code = tl.where(value < 5.0, code, 7)
    return code


@triton.jit
def pack_nvfp4_kernel(
    x,
    packed,
    raw_scales,
    scales,
    global_scale,
    k: tl.constexpr,
    block_size: tl.constexpr,
    column_tiles: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    pairs = tl.arange(0, block_size // 2)
    offsets = block * block_size + pairs * 2
    raw_scale = tl.load(raw_scales + row * (k // block_size) + block)
    global_value = tl.load(global_scale)
    value_0 = tl.load(x + row * k + offsets).to(tl.float32) / raw_scale
    value_1 = tl.load(x + row * k + offsets + 1).to(tl.float32) / raw_scale
    code_0 = fp4_code(value_0) | tl.where(value_0 < 0, 8, 0)
    code_1 = fp4_code(value_1) | tl.where(value_1 < 0, 8, 0)
    tl.store(
        packed + row * (k // 2) + block * (block_size // 2) + pairs,
        code_0 | (code_1 << 4),
    )
    scale_offset = (
        512 * ((row // 128) * column_tiles + block // 4)
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + block % 4
    )
    tl.store(scales + scale_offset, raw_scale / global_value)


@triton.jit
def pack_nvfp4_fixed_kernel(
    x,
    packed,
    scales,
    global_scale,
    k: tl.constexpr,
    block_size: tl.constexpr,
    column_tiles: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    pairs = tl.arange(0, block_size // 2)
    offsets = block * block_size + pairs * 2
    global_value = tl.load(global_scale)
    value_0 = tl.load(x + row * k + offsets).to(tl.float32)
    value_1 = tl.load(x + row * k + offsets + 1).to(tl.float32)
    raw_scale = tl.maximum(
        tl.maximum(tl.max(tl.abs(value_0), axis=0), tl.max(tl.abs(value_1), axis=0)) / 6.0,
        global_value / 512.0,
    )
    raw_scale = tl.minimum(raw_scale, global_value * 448.0)
    value_0 /= raw_scale
    value_1 /= raw_scale
    code_0 = fp4_code(value_0) | tl.where(value_0 < 0, 8, 0)
    code_1 = fp4_code(value_1) | tl.where(value_1 < 0, 8, 0)
    tl.store(
        packed + row * (k // 2) + block * (block_size // 2) + pairs,
        code_0 | (code_1 << 4),
    )
    scale_offset = (
        512 * ((row // 128) * column_tiles + block // 4)
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + block % 4
    )
    tl.store(scales + scale_offset, raw_scale / global_value)


def quant_nvfp4(x: torch.Tensor, block_size: int = 16):
    assert x.is_cuda and x.is_contiguous() and x.ndim == 2
    assert x.shape[1] % block_size == 0
    rows, k = x.shape
    blocks = k // block_size
    raw_scales = torch.empty(rows, blocks, dtype=torch.float32, device=x.device)
    block_scales_kernel[(rows, blocks)](
        x,
        raw_scales,
        k,
        block_size,
    )
    global_scale = raw_scales.amax().clamp_min(1e-12) / 448.0
    packed = torch.empty(rows, k // 2, dtype=torch.uint8, device=x.device)
    row_tiles = triton.cdiv(rows, 128)
    column_tiles = triton.cdiv(blocks, 4)
    scales = torch.zeros(
        row_tiles * column_tiles * 512,
        dtype=torch.float8_e4m3fn,
        device=x.device,
    )
    pack_nvfp4_kernel[(rows, blocks)](
        x,
        packed,
        raw_scales,
        scales,
        global_scale,
        k,
        block_size,
        column_tiles,
    )
    return packed.view(torch.float4_e2m1fn_x2), scales, global_scale


def quant_nvfp4_fixed(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    block_size: int = 16,
):
    assert x.is_cuda and x.is_contiguous() and x.ndim == 2
    assert x.shape[1] % block_size == 0
    rows, k = x.shape
    blocks = k // block_size
    row_tiles = triton.cdiv(rows, 128)
    column_tiles = triton.cdiv(blocks, 4)
    packed = torch.empty(rows, k // 2, dtype=torch.uint8, device=x.device)
    scales = torch.zeros(
        row_tiles * column_tiles * 512,
        dtype=torch.float8_e4m3fn,
        device=x.device,
    )
    pack_nvfp4_fixed_kernel[(rows, blocks)](
        x,
        packed,
        scales,
        global_scale,
        k,
        block_size,
        column_tiles,
    )
    return packed.view(torch.float4_e2m1fn_x2), scales, global_scale


@triton.jit
def grouped_nvfp4_mm_kernel(
    x,
    weight,
    x_scales,
    weight_scales,
    x_global_scales,
    weight_global_scales,
    expert_ids,
    group_sizes,
    output,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    x_scale_size: tl.constexpr,
    weight_scale_size: tl.constexpr,
    scale_columns: tl.constexpr,
    use_indices: tl.constexpr,
):
    group = tl.program_id(0)
    if use_indices:
        expert = tl.load(expert_ids + group)
        group_size = tl.load(group_sizes + group)
    else:
        expert = group
        group_size = m
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    offsets_m = pid_m * block_m + tl.arange(0, block_m)
    offsets_n = pid_n * block_n + tl.arange(0, block_n)
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for block in range(0, tl.cdiv(k, block_k)):
        offsets_k = block * (block_k // 2) + tl.arange(0, block_k // 2)
        x_values = tl.load(
            x + group * m * (k // 2) + offsets_m[:, None] * (k // 2) + offsets_k[None, :],
            mask=(offsets_m[:, None] < group_size) & (offsets_k[None, :] < k // 2),
            other=0,
        )
        weight_values = tl.load(
            weight + expert * (k // 2) * n + offsets_k[:, None] * n + offsets_n[None, :],
            mask=offsets_k[:, None] < k // 2,
            other=0,
        )
        scale_k = block * (block_k // 16) + tl.arange(0, block_k // 16)
        x_scale_offsets = (
            512 * ((offsets_m[:, None] // 128) * scale_columns + scale_k[None, :] // 4)
            + (offsets_m[:, None] % 32) * 16
            + ((offsets_m[:, None] % 128) // 32) * 4
            + scale_k[None, :] % 4
        )
        weight_scale_offsets = (
            512 * ((offsets_n[:, None] // 128) * scale_columns + scale_k[None, :] // 4)
            + (offsets_n[:, None] % 32) * 16
            + ((offsets_n[:, None] % 128) // 32) * 4
            + scale_k[None, :] % 4
        )
        x_scale_values = tl.load(
            x_scales + group * x_scale_size + x_scale_offsets,
            mask=(offsets_m[:, None] < group_size) & (scale_k[None, :] < k // 16),
            other=0.0,
        )
        weight_scale_values = tl.load(
            weight_scales + expert * weight_scale_size + weight_scale_offsets,
            mask=scale_k[None, :] < k // 16,
            other=0.0,
        )
        accumulator = tl.dot_scaled(
            x_values,
            x_scale_values,
            "e2m1",
            weight_values,
            weight_scale_values,
            "e2m1",
            accumulator,
        )
    accumulator *= tl.load(x_global_scales)
    accumulator *= tl.load(weight_global_scales + expert)
    tl.store(
        output + group * m * n + offsets_m[:, None] * n + offsets_n[None, :],
        accumulator,
        mask=offsets_m[:, None] < group_size,
    )


def grouped_nvfp4_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    x_global_scales: torch.Tensor,
    weight_global_scales: torch.Tensor,
    expert_ids: torch.Tensor | None = None,
    group_sizes: torch.Tensor | None = None,
):
    groups, m, packed_k = x.shape
    k = packed_k * 2
    n = weight.shape[-1]
    block_m = block_n = 128
    block_k = 256
    scale_columns = triton.cdiv(k // 16, 4)
    x_scale_size = triton.cdiv(m, 128) * scale_columns * 512
    weight_scale_size = triton.cdiv(n, 128) * scale_columns * 512
    output = torch.zeros(groups, m, n, dtype=torch.bfloat16, device=x.device)
    use_indices = expert_ids is not None
    if expert_ids is None:
        expert_ids = weight_global_scales
        group_sizes = weight_global_scales
    grouped_nvfp4_mm_kernel[
        (groups, triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    ](
        x.view(torch.uint8),
        weight.view(torch.uint8),
        x_scales,
        weight_scales,
        x_global_scales,
        weight_global_scales,
        expert_ids,
        group_sizes,
        output,
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        x_scale_size,
        weight_scale_size,
        scale_columns,
        use_indices,
        num_warps=8,
    )
    return output


def nvfp4_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    x_global_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
):
    return nvfp4_mm_packed(
        x,
        weight.t(),
        x_scales,
        weight_scales,
        x_global_scale,
        weight_global_scale,
    )


def nvfp4_mm_packed(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    x_global_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
):
    return F.scaled_mm(
        x,
        weight,
        [x_scales, x_global_scale],
        [ScalingType.BlockWise1x16, ScalingType.TensorWise],
        [weight_scales, weight_global_scale],
        [ScalingType.BlockWise1x16, ScalingType.TensorWise],
        [SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
        [SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
        output_dtype=torch.bfloat16,
    )
