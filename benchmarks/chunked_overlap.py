import json
import os

import torch
import torch.distributed as dist

from nanovllm.layers.linear import FP8RowParallelLinear


def rank_max(value):
    result = torch.tensor(value, device="cuda", dtype=torch.float64)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return result.item()


def serial(layer, x):
    output = layer.apply(x)
    dist.all_reduce(output)
    return output


def chunked(layer, x, chunks, communication_stream):
    outputs = []
    works = []
    for x_chunk in x.chunk(chunks):
        output = layer.apply(x_chunk)
        ready = torch.cuda.Event()
        ready.record()
        with torch.cuda.stream(communication_stream):
            communication_stream.wait_event(ready)
            works.append(dist.all_reduce(output, async_op=True))
        outputs.append(output)
    for work in works:
        work.wait()
    torch.cuda.current_stream().wait_stream(communication_stream)
    return torch.cat(outputs)


def time(run, warmup, iterations):
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        output = run()
    end.record()
    end.synchronize()
    latency_ms = rank_max(start.elapsed_time(end) / iterations)
    return output, latency_ms


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    torch.manual_seed(0)
    layer = FP8RowParallelLinear(6144, 3072, reduce_results=False).cuda()
    layer.weight.data.copy_(torch.randn_like(layer.weight, dtype=torch.bfloat16).to(layer.weight.dtype))
    layer.weight_scale_inv.data.fill_(1.0 / 1536 ** 0.5)
    x = torch.randn(2048, 1536, device="cuda", dtype=torch.bfloat16)
    communication_stream = torch.cuda.Stream()
    reference, baseline_ms = time(lambda: serial(layer, x), 20, 100)
    results = []
    for chunks in [2, 4, 8, 16]:
        output, latency_ms = time(
            lambda: chunked(layer, x, chunks, communication_stream),
            20,
            100,
        )
        error = (output.float() - reference.float()).abs()
        results.append(
            {
                "chunks": chunks,
                "latency_ms": latency_ms,
                "speedup": baseline_ms / latency_ms,
                "mean_error": error.mean().item(),
                "max_error": error.max().item(),
            }
        )
    if rank == 0:
        print(json.dumps({"chunks": 1, "latency_ms": baseline_ms, "speedup": 1.0}))
        for result in results:
            print(json.dumps(result, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
