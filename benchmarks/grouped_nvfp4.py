import json
import os

import torch
import torch.distributed as dist

from nanovllm.layers.nvfp4 import grouped_nvfp4_mm, nvfp4_mm, quant_nvfp4, quant_nvfp4_fixed


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


def benchmark(tokens, k, n, groups=8):
    torch.manual_seed(0)
    global_scale = torch.tensor(1.0 / 448.0, device="cuda")
    x = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16)
    x, x_scales, _ = quant_nvfp4_fixed(x, global_scale)
    weights = []
    weight_scales = []
    weight_globals = []
    for _ in range(groups):
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k ** 0.5
        weight, scales, tensor_scale = quant_nvfp4(weight)
        weights.append(weight)
        weight_scales.append(scales)
        weight_globals.append(tensor_scale)
    grouped_x = x.unsqueeze(0).expand(groups, -1, -1).contiguous()
    grouped_x_scales = x_scales.unsqueeze(0).expand(groups, -1).contiguous()
    grouped_weight = torch.stack([weight.t() for weight in weights])
    grouped_weight_scales = torch.stack(weight_scales)
    grouped_weight_globals = torch.stack(weight_globals).reshape(groups)
    expert_ids = torch.arange(groups, dtype=torch.int32, device="cuda")
    group_sizes = torch.full((groups,), tokens, dtype=torch.int32, device="cuda")

    def serial():
        return torch.stack(
            [
                nvfp4_mm(
                    x,
                    weights[index],
                    x_scales,
                    weight_scales[index],
                    global_scale,
                    weight_globals[index],
                )
                for index in range(groups)
            ]
        )

    def grouped():
        return grouped_nvfp4_mm(
            grouped_x,
            grouped_weight,
            grouped_x_scales,
            grouped_weight_scales,
            global_scale,
            grouped_weight_globals,
            expert_ids,
            group_sizes,
        )

    reference, serial_ms = time(serial)
    output, grouped_ms = time(grouped)
    error = (output.float() - reference.float()).abs()
    return {
        "tokens_per_expert": tokens,
        "groups": groups,
        "k": k,
        "n": n,
        "serial_ms": serial_ms,
        "grouped_ms": grouped_ms,
        "speedup": serial_ms / grouped_ms,
        "mean_error": error.mean().item(),
        "max_error": error.max().item(),
    }


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    results = []
    for tokens in [1, 16, 128, 512, 2048]:
        results.append(benchmark(tokens, 3072, 2048, 1))
        results.append(benchmark(tokens, 1536, 3072, 1))
    if rank == 0:
        for result in results:
            print(json.dumps(result, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
