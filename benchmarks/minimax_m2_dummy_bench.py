import argparse
import json
import math
import os
import platform
import socket
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import PretrainedConfig

from nanovllm.layers.embed_head import VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import LinearBase, NVFP4GroupedLinear, NVFP4LinearBase
from nanovllm.models.minimax_m2 import MiniMaxM2ForCausalLM
from nanovllm.utils.context import reset_context, set_context
from nanovllm.utils.loader import initialize_dummy_weights


LABEL = "MiniMax M2 faithful dummy benchmark"
ALL_REDUCE = dist.all_reduce
ALL_REDUCE_EVENTS = []


def timed_all_reduce(tensor, *args, **kwargs):
    result = ALL_REDUCE(tensor, *args, **kwargs)
    ALL_REDUCE_EVENTS.append(tensor.numel() * tensor.element_size())
    return result


def parse_int_list(value: str) -> list[int]:
    if value == "none":
        return []
    values = [int(item) for item in value.split(",")]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prefill-batches", type=parse_int_list, default=[1])
    parser.add_argument("--prefill-context", type=int, default=2048)
    parser.add_argument("--decode-batches", type=parse_int_list, default=[1])
    parser.add_argument("--decode-context", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--dummy-seed", type=int)
    return parser.parse_args()


def initialize_seeded_dummy_weights(model, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (NVFP4LinearBase, NVFP4GroupedLinear)):
                module.weight.view(torch.uint8).random_(0, 256)
                module.weight_scale_inv.fill_(1)
                module.weight_global_scale.fill_(
                    1.0 / (4 * math.sqrt(module.weight.shape[1] * 2))
                )
            elif isinstance(module, LinearBase):
                module.weight.normal_(0.0, 1.0 / math.sqrt(module.weight.shape[1]))
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, VocabParallelEmbedding):
                module.weight.normal_(0.0, 1.0 / math.sqrt(module.weight.shape[1]))
            elif isinstance(module, RMSNorm):
                module.weight.fill_(1.0)


def rank_max(value: float) -> float:
    result = torch.tensor(value, dtype=torch.float64, device="cuda")
    ALL_REDUCE(result, op=dist.ReduceOp.MAX)
    return result.item()


def timed_forward(run, tokens: int, warmup: int, iterations: int) -> dict:
    for _ in range(warmup):
        output = run()
    torch.cuda.synchronize()
    dist.barrier()
    ALL_REDUCE_EVENTS.clear()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        output = run()
    end.record()
    end.synchronize()
    all_reduce_events = list(ALL_REDUCE_EVENTS)
    elapsed_ms = rank_max(start.elapsed_time(end) / iterations)
    if output is not None:
        assert torch.isfinite(output).all()
    return {
        "latency_ms": elapsed_ms,
        "tokens_per_second": tokens * 1000.0 / elapsed_ms,
        "all_reduce_calls": len(all_reduce_events) // iterations,
        "all_reduce_bytes": sum(all_reduce_events) // iterations,
    }


def prefill_case(
    model: MiniMaxM2ForCausalLM,
    batch: int,
    context: int,
    warmup: int,
    iterations: int,
) -> dict:
    input_ids = torch.zeros(batch * context, dtype=torch.int64, device="cuda")
    positions = torch.arange(context, dtype=torch.int64, device="cuda").repeat(batch)
    cu_seqlens = torch.arange(
        0,
        (batch + 1) * context,
        context,
        dtype=torch.int32,
        device="cuda",
    )
    set_context(
        True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=context,
        max_seqlen_k=context,
    )

    def run():
        return model.compute_logits(model(input_ids, positions))

    result = timed_forward(run, batch * context, warmup, iterations)
    reset_context()
    result.update(
        {
            "phase": "prefill",
            "batch": batch,
            "prompt_tokens_per_sequence": context,
            "tokens_per_iteration": batch * context,
        }
    )
    return result


def attach_zero_kv_cache(
    model: MiniMaxM2ForCausalLM,
    config: PretrainedConfig,
    batch: int,
    context: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    world_size = dist.get_world_size()
    blocks_per_sequence = math.ceil(context / block_size)
    num_blocks = batch * blocks_per_sequence
    num_kv_heads = config.num_key_value_heads // world_size
    kv_cache = torch.zeros(
        2,
        config.num_hidden_layers,
        num_blocks,
        block_size,
        num_kv_heads,
        config.head_dim,
        dtype=config.dtype,
        device="cuda",
    )
    layer_id = 0
    for module in model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = kv_cache[0, layer_id]
            module.v_cache = kv_cache[1, layer_id]
            layer_id += 1
    assert layer_id == config.num_hidden_layers
    block_tables = torch.arange(
        num_blocks,
        dtype=torch.int32,
        device="cuda",
    ).view(batch, blocks_per_sequence)
    slot_mapping = (
        block_tables[:, -1] * block_size + (context - 1) % block_size
    ).to(torch.int32)
    return kv_cache, block_tables, slot_mapping


def decode_case(
    model: MiniMaxM2ForCausalLM,
    config: PretrainedConfig,
    batch: int,
    context: int,
    block_size: int,
    warmup: int,
    iterations: int,
) -> dict:
    kv_cache, block_tables, slot_mapping = attach_zero_kv_cache(
        model,
        config,
        batch,
        context,
        block_size,
    )
    input_ids = torch.zeros(batch, dtype=torch.int64, device="cuda")
    positions = torch.full(
        (batch,),
        context - 1,
        dtype=torch.int64,
        device="cuda",
    )
    context_lens = torch.full(
        (batch,),
        context,
        dtype=torch.int32,
        device="cuda",
    )
    set_context(
        False,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
    )

    def run():
        return model.compute_logits(model(input_ids, positions))

    result = timed_forward(run, batch, warmup, iterations)
    reset_context()
    result.update(
        {
            "phase": "decode",
            "batch": batch,
            "total_sequence_tokens": context,
            "cached_tokens_per_sequence": context - 1,
            "query_tokens_per_sequence": 1,
            "tokens_per_iteration": batch,
            "kv_cache_bytes_per_rank": kv_cache.numel() * kv_cache.element_size(),
        }
    )
    del kv_cache
    for module in model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = module.v_cache = torch.tensor([], device="cuda")
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if args.prefill_context < 1 or args.decode_context < 1:
        raise ValueError("context lengths must be positive")
    if args.block_size < 1 or args.warmup < 0 or args.iterations < 1:
        raise ValueError("invalid block size, warmup, or iteration count")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    with args.config.open() as config_file:
        config_dict = json.load(config_file)
    config = PretrainedConfig.from_dict(config_dict)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(config.dtype)
    torch.set_default_device("cuda")

    model = MiniMaxM2ForCausalLM(config)
    if args.dummy_seed is None:
        initialize_dummy_weights(model)
    else:
        initialize_seeded_dummy_weights(model, args.dummy_seed)
    model.eval()
    dist.all_reduce = timed_all_reduce
    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "metadata",
                    "label": LABEL,
                    "hostname": socket.gethostname(),
                    "platform": platform.platform(),
                    "world_size": dist.get_world_size(),
                    "torch": torch.__version__,
                    "torch_cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(),
                    "compute_capability": list(torch.cuda.get_device_capability()),
                    "config_path": str(args.config),
                    "config": config_dict,
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                    "dummy_seed": args.dummy_seed,
                    "block_size": args.block_size,
                    "parameters_per_rank": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "allocated_bytes_per_rank": torch.cuda.memory_allocated(),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    with torch.inference_mode():
        for batch in args.prefill_batches:
            result = prefill_case(
                model,
                batch,
                args.prefill_context,
                args.warmup,
                args.iterations,
            )
            if rank == 0:
                print(json.dumps({"event": "result", **result}, sort_keys=True), flush=True)

        for batch in args.decode_batches:
            result = decode_case(
                model,
                config,
                batch,
                args.decode_context,
                args.block_size,
                args.warmup,
                args.iterations,
            )
            if rank == 0:
                print(json.dumps({"event": "result", **result}, sort_keys=True), flush=True)

    dist.barrier()
    dist.destroy_process_group()
    torch.set_default_device("cpu")
    torch.set_default_dtype(default_dtype)


if __name__ == "__main__":
    main()
