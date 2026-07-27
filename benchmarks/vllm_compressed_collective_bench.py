import json
import os

import torch
import torch.distributed as dist

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.distributed.compressed_all_reduce import (
    get_stats,
    maybe_compressed_all_reduce,
    reset_stats,
)


def rank_max(value: float) -> float:
    output = torch.tensor(value, dtype=torch.float64, device="cuda")
    dist.all_reduce(output, op=dist.ReduceOp.MAX)
    return output.item()


def time(run) -> float:
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        run()
    end.record()
    end.synchronize()
    return rank_max(start.elapsed_time(end) / 100)


def benchmark(tokens: int) -> dict:
    group = get_tp_group()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(1234 + dist.get_rank())
    source = torch.randn(
        tokens,
        3072,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    reference = source.clone()
    dist.all_reduce(reference)
    output = maybe_compressed_all_reduce(source, group)
    selected = output is not None
    if output is None:
        output = group._all_reduce_out_place(source)
    error = (output.float() - reference.float()).abs()

    baseline_ms = time(lambda: group._all_reduce_out_place(source))
    reset_stats()

    def compressed():
        result = maybe_compressed_all_reduce(source, group)
        return group._all_reduce_out_place(source) if result is None else result

    compressed_ms = time(compressed)
    return {
        "tokens": tokens,
        "input_bytes": source.nbytes,
        "selected": selected,
        "baseline_ms": baseline_ms,
        "compressed_ms": compressed_ms,
        "speedup": baseline_ms / compressed_ms,
        "mean_relative_error": (
            error.mean() / reference.float().abs().mean().clamp_min(1e-12)
        ).item(),
        "max_relative_error": (
            error.max() / reference.float().abs().max().clamp_min(1e-12)
        ).item(),
        "stats": get_stats(),
    }


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method="env://",
        local_rank=local_rank,
    )
    vllm_config = VllmConfig()
    vllm_config.parallel_config.tensor_parallel_size = world_size
    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(tensor_model_parallel_size=world_size)
    results = [benchmark(tokens) for tokens in (1, 16, 128, 512, 2048)]
    if rank == 0:
        for result in results:
            print(json.dumps(result, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
