import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from nanovllm.layers.fp8 import all_reduce


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


def rms_norm(x):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + 1e-6)).to(x.dtype)


def benchmark(tokens):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    hidden_size = 3072
    experts = 256
    shard_size = hidden_size // world_size
    torch.manual_seed(0)
    partial = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    rank_major = (
        partial.view(tokens, world_size, shard_size)
        .permute(1, 0, 2)
        .contiguous()
        .view(world_size * tokens, shard_size)
    )
    gate = torch.randn(experts, hidden_size, device="cuda", dtype=torch.bfloat16)
    gate_shard = gate[:, rank * shard_size:(rank + 1) * shard_size].contiguous()

    def reference():
        hidden = partial.clone()
        dist.all_reduce(hidden)
        return F.linear(rms_norm(hidden), gate)

    def compressed():
        hidden = partial.clone()
        hidden = all_reduce(hidden)
        return F.linear(rms_norm(hidden), gate)

    def sharded():
        hidden = torch.empty(
            tokens,
            shard_size,
            dtype=partial.dtype,
            device=partial.device,
        )
        dist.reduce_scatter_tensor(hidden, rank_major)
        square_sum = hidden.float().pow(2).sum(-1, keepdim=True)
        dist.all_reduce(square_sum)
        hidden = hidden.float() * torch.rsqrt(
            square_sum / hidden_size + 1e-6
        )
        logits = F.linear(hidden.to(partial.dtype), gate_shard)
        dist.all_reduce(logits)
        return logits

    exact = reference()
    current, current_ms = time(compressed)
    output, sharded_ms = time(sharded)
    current_error = (current.float() - exact.float()).abs()
    error = (output.float() - exact.float()).abs()
    return {
        "tokens": tokens,
        "current_ms": current_ms,
        "sharded_ms": sharded_ms,
        "speedup": current_ms / sharded_ms,
        "current_mean_error": (
            current_error.mean() / exact.float().abs().mean().clamp_min(1e-12)
        ).item(),
        "sharded_mean_error": (
            error.mean() / exact.float().abs().mean().clamp_min(1e-12)
        ).item(),
        "sharded_max_error": (
            error.max() / exact.float().abs().max().clamp_min(1e-12)
        ).item(),
        "full_hidden_bytes": partial.nbytes,
        "router_bytes": tokens * experts * partial.element_size(),
        "norm_stat_bytes": tokens * 4,
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
