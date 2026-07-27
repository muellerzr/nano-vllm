import json
import os

import torch
import torch.distributed as dist


def rank_max(value):
    result = torch.tensor(value, device="cuda", dtype=torch.float64)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return result.item()


def benchmark(tokens, warmup, iterations):
    tensor = torch.zeros(tokens, 3072, device="cuda", dtype=torch.bfloat16)
    for _ in range(warmup):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        dist.all_reduce(tensor)
    end.record()
    end.synchronize()
    latency_ms = rank_max(start.elapsed_time(end) / iterations)
    return {
        "tokens": tokens,
        "bytes": tensor.numel() * tensor.element_size(),
        "latency_ms": latency_ms,
        "algorithm_gbps": tensor.numel() * tensor.element_size() / latency_ms / 1e6,
    }


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    results = [benchmark(tokens, 20, 100) for tokens in [1, 16, 128, 2048]]
    if rank == 0:
        for result in results:
            print(json.dumps(result, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
