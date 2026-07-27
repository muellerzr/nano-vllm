import torch
import triton
import triton.language as tl


@triton.jit
def _quant_kernel(x, q, scales, k: tl.constexpr, group: tl.constexpr):
    row = tl.program_id(0)
    col_group = tl.program_id(1)
    cols = col_group * group + tl.arange(0, group)
    values = tl.load(x + row * k + cols).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values), axis=0), 1e-10) / 448.0
    tl.store(q + row * k + cols, values / scale)
    tl.store(scales + row * (k // group) + col_group, scale)


def per_token_group_quant_fp8(x: torch.Tensor, group_size: int):
    assert x.is_cuda and x.is_contiguous() and x.shape[-1] % group_size == 0
    x = x.view(-1, x.shape[-1])
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(x.shape[0], x.shape[1] // group_size, device=x.device, dtype=torch.float32)
    _quant_kernel[(x.shape[0], x.shape[1] // group_size)](x, q, scales, x.shape[1], group_size)
    return q.view_as(x), scales


@triton.jit
def _scaled_mm_kernel(
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
    rows = pid_m * block_m + tl.arange(0, block_m)
    cols = pid_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for block in range(0, tl.cdiv(k, block_k)):
        x_values = tl.load(x + rows[:, None] * k + block * block_k + offsets_k[None, :], mask=rows[:, None] < m, other=0.0)
        weight_values = tl.load(weight + cols[:, None] * k + block * block_k + offsets_k[None, :], mask=cols[:, None] < n, other=0.0)
        x_scale = tl.load(x_scales + rows * (k // block_k) + block, mask=rows < m, other=0.0)
        weight_scale = tl.load(weight_scales + (cols // block_n) * (k // block_k) + block, mask=cols < n, other=0.0)
        accumulator += tl.dot(x_values, tl.trans(weight_values)) * x_scale[:, None] * weight_scale[None, :]
    tl.store(output + rows[:, None] * n + cols[None, :], accumulator, mask=(rows[:, None] < m) & (cols[None, :] < n))


def w8a8_triton_block_scaled_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous() and weight.is_contiguous()
    m, k = x.shape
    n = weight.shape[0]
    block_n, block_k = block_size
    block_m = 16 if m < 64 else 64
    output = torch.empty((m, n), dtype=output_dtype, device=x.device)
    _scaled_mm_kernel[(triton.cdiv(m, block_m), triton.cdiv(n, block_n))](
        x, weight, x_scales, weight_scales, output, m, n, k, block_m, block_n, block_k,
    )
    return output
