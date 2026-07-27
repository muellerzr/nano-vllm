import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from nanovllm.layers.fp8 import block_scaled_mm, quant_fp8
from nanovllm.layers.nvfp4 import nvfp4_mm, quant_nvfp4, quant_nvfp4_fixed


def rank_max(value):
    result = torch.tensor(value, device="cuda", dtype=torch.float64)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return result.item()


def time(run):
    for _ in range(20):
        run()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        output = run()
    end.record()
    end.synchronize()
    return output, rank_max(start.elapsed_time(end) / 100)


def benchmark(tokens):
    torch.manual_seed(0)
    x = torch.randn(tokens, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(3072, 1536, device="cuda", dtype=torch.bfloat16) / 1536 ** 0.5
    weight_fp8, weight_fp8_scales = quant_fp8(weight)
    weight_fp4, weight_fp4_scales, weight_fp4_global = quant_nvfp4(weight)
    fixed_global = torch.tensor(1.0 / 448.0, device="cuda")
    x_fp8, x_fp8_scales = quant_fp8(x)
    x_fp4, x_fp4_scales, x_fp4_global = quant_nvfp4(x)
    x_fp4_fixed, x_fp4_fixed_scales, _ = quant_nvfp4_fixed(x, fixed_global)

    def fp8_mm():
        return block_scaled_mm(
            x_fp8,
            weight_fp8,
            x_fp8_scales,
            weight_fp8_scales,
        )

    def fp4_mm():
        return nvfp4_mm(
            x_fp4,
            weight_fp4,
            x_fp4_scales,
            weight_fp4_scales,
            x_fp4_global,
            weight_fp4_global,
        )

    def fp4_fixed_mm():
        return nvfp4_mm(
            x_fp4_fixed,
            weight_fp4,
            x_fp4_fixed_scales,
            weight_fp4_scales,
            fixed_global,
            weight_fp4_global,
        )

    _, fp8_quant_ms = time(lambda: quant_fp8(x))
    _, fp4_quant_ms = time(lambda: quant_nvfp4(x))
    _, fp4_fixed_quant_ms = time(lambda: quant_nvfp4_fixed(x, fixed_global))
    output_fp8, fp8_mm_ms = time(fp8_mm)
    output_fp4, fp4_mm_ms = time(fp4_mm)
    output_fp4_fixed, fp4_fixed_mm_ms = time(fp4_fixed_mm)
    repeat_x, repeat_scales, _ = quant_nvfp4_fixed(x, fixed_global)
    repeat_output = nvfp4_mm(
        repeat_x,
        weight_fp4,
        repeat_scales,
        weight_fp4_scales,
        fixed_global,
        weight_fp4_global,
    )
    reference = F.linear(x, weight)
    error = (output_fp4.float() - reference.float()).abs()
    fixed_error = (output_fp4_fixed.float() - reference.float()).abs()
    return {
        "tokens": tokens,
        "fp8_quant_ms": fp8_quant_ms,
        "nvfp4_quant_ms": fp4_quant_ms,
        "nvfp4_fixed_quant_ms": fp4_fixed_quant_ms,
        "fp8_mm_ms": fp8_mm_ms,
        "nvfp4_mm_ms": fp4_mm_ms,
        "nvfp4_fixed_mm_ms": fp4_fixed_mm_ms,
        "mm_speedup": fp8_mm_ms / fp4_mm_ms,
        "fp8_total_ms": fp8_quant_ms + fp8_mm_ms,
        "nvfp4_total_ms": fp4_quant_ms + fp4_mm_ms,
        "nvfp4_fixed_total_ms": fp4_fixed_quant_ms + fp4_fixed_mm_ms,
        "total_speedup": (
            (fp8_quant_ms + fp8_mm_ms) / (fp4_quant_ms + fp4_mm_ms)
        ),
        "mean_error": (
            error.mean() / reference.float().abs().mean().clamp_min(1e-12)
        ).item(),
        "max_error": (
            error.max() / reference.float().abs().max().clamp_min(1e-12)
        ).item(),
        "fixed_mean_error": (
            fixed_error.mean() / reference.float().abs().mean().clamp_min(1e-12)
        ).item(),
        "fixed_max_error": (
            fixed_error.max() / reference.float().abs().max().clamp_min(1e-12)
        ).item(),
        "fixed_repeat_max_error": (
            repeat_output.float() - output_fp4_fixed.float()
        ).abs().max().item(),
        "fp8_weight_bytes": weight_fp8.nbytes + weight_fp8_scales.nbytes,
        "nvfp4_weight_bytes": (
            weight_fp4.nbytes + weight_fp4_scales.nbytes + weight_fp4_global.nbytes
        ),
    }


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    results = [benchmark(tokens) for tokens in [1, 16, 128, 512, 2048]]
    if rank == 0:
        for result in results:
            print(json.dumps(result, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
