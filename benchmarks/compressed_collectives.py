import json
import os

import torch
import torch.distributed as dist

from nanovllm.layers.fp8 import fp8_all_reduce


def rank_max(value):
    result = torch.tensor(value, device="cuda", dtype=torch.float64)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return result.item()


def time(run, warmup, iterations):
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        run()
    end.record()
    end.synchronize()
    return rank_max(start.elapsed_time(end) / iterations)


def benchmark(tokens):
    source = torch.zeros(tokens, 3072, device="cuda", dtype=torch.bfloat16)
    baseline = source.clone()
    baseline_ms = time(lambda: dist.all_reduce(baseline), 20, 100)
    compressed_ms = time(lambda: fp8_all_reduce(source), 20, 100)
    torch.manual_seed(0)
    source = torch.randn_like(source)
    reference = source.clone()
    dist.all_reduce(reference)
    output = fp8_all_reduce(source)
    error = (output.float() - reference.float()).abs()
    mean_error = error.mean() / reference.float().abs().mean().clamp_min(1e-12)
    max_error = error.max() / reference.float().abs().max().clamp_min(1e-12)
    return {
        "tokens": tokens,
        "bytes": source.numel() * source.element_size(),
        "baseline_ms": baseline_ms,
        "compressed_ms": compressed_ms,
        "speedup": baseline_ms / compressed_ms,
        "mean_error": mean_error.item(),
        "max_error": max_error.item(),
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
