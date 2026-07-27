#!/usr/bin/env python3
"""Reproduce stock vs FP8-collective NanoVLLM timings with synthetic weights."""

import argparse
import json
import math
import os
import platform
import socket
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import PretrainedConfig

from nanovllm.layers.compressed_collective import all_reduce, reset_stats, stats
from nanovllm.models.minimax_m2 import MiniMaxM2ForCausalLM
from nanovllm.utils.context import reset_context, set_context


CASES = (("prefill", 32, 16), ("decode", 1, 16), ("prefill", 4, 512), ("decode", 1, 2048))


def rank_max(value: float) -> float:
    result = torch.tensor(value, dtype=torch.float64, device="cuda")
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return result.item()


def source_revision() -> str:
    root = Path(__file__).resolve().parents[1]
    return subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=root, text=True).strip()


def seed_weights(model: torch.nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("weight_scale_inv"):
                parameter.fill_(1)
            elif parameter.dtype == torch.float8_e4m3fn:
                parameter.copy_(torch.randn(parameter.shape, device="cuda", dtype=torch.bfloat16).to(parameter.dtype))
            elif parameter.ndim == 1 and name.endswith(".weight"):
                parameter.fill_(1)
            else:
                parameter.normal_(0, 1 / math.sqrt(max(1, parameter.shape[-1])))


def time_forward(run, tokens: int, warmup: int, iterations: int) -> dict:
    for _ in range(warmup):
        output = run()
    torch.cuda.synchronize()
    dist.barrier()
    reset_stats()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iterations):
        output = run()
    end.record()
    end.synchronize()
    if output is not None:
        assert torch.isfinite(output).all()
    latency_ms = rank_max(start.elapsed_time(end) / iterations)
    return {
        "latency_ms": latency_ms,
        "tokens_per_second": tokens * 1000 / latency_ms,
        "collective": stats(),
    }


def attach_kv_cache(model, config, batch: int, context: int, block_size: int):
    world = dist.get_world_size()
    blocks_per_sequence = math.ceil(context / block_size)
    cache = torch.zeros(
        2,
        config.num_hidden_layers,
        batch * blocks_per_sequence,
        block_size,
        config.num_key_value_heads // world,
        config.head_dim,
        dtype=config.dtype,
        device="cuda",
    )
    layer = 0
    for module in model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache, module.v_cache = cache[0, layer], cache[1, layer]
            layer += 1
    tables = torch.arange(cache.shape[2], dtype=torch.int32, device="cuda").view(batch, blocks_per_sequence)
    slots = (tables[:, -1] * block_size + (context - 1) % block_size).to(torch.int32)
    return cache, tables, slots


def case(model, config, phase: str, batch: int, context: int, args) -> dict:
    if phase == "prefill":
        input_ids = torch.zeros(batch * context, dtype=torch.int64, device="cuda")
        positions = torch.arange(context, dtype=torch.int64, device="cuda").repeat(batch)
        cu = torch.arange(0, (batch + 1) * context, context, dtype=torch.int32, device="cuda")
        set_context(True, cu, cu, context, context)
        cleanup = None
        tokens = batch * context
    else:
        cache, tables, slots = attach_kv_cache(model, config, batch, context, args.block_size)
        input_ids = torch.zeros(batch, dtype=torch.int64, device="cuda")
        positions = torch.full((batch,), context - 1, dtype=torch.int64, device="cuda")
        lengths = torch.full((batch,), context, dtype=torch.int32, device="cuda")
        set_context(False, slot_mapping=slots, context_lens=lengths, block_tables=tables)
        cleanup = cache
        tokens = batch

    def run():
        return model.compute_logits(model(input_ids, positions))

    result = time_forward(run, tokens, args.warmup, args.iterations)
    reset_context()
    if cleanup is not None:
        for module in model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = module.v_cache = torch.tensor([], device="cuda")
        del cleanup
        torch.cuda.empty_cache()
    return {"phase": phase, "batch": batch, "context": context, **result}


def validate_collective(args) -> dict:
    elements = math.ceil(args.min_bytes / 2 / 128) * 128
    torch.manual_seed(args.seed)
    source = torch.randn(elements, dtype=torch.bfloat16, device="cuda")
    reference = source.clone()
    dist.all_reduce(reference)
    os.environ["NANOVLLM_FP8_ALL_REDUCE"] = "1"
    candidate = source.clone()
    all_reduce(candidate)
    error = (candidate.float() - reference.float()).abs()
    mean_relative = error.mean() / reference.float().abs().mean().clamp_min(1e-12)
    max_relative = error.max() / reference.float().abs().max().clamp_min(1e-12)
    return {"bytes": source.numel() * source.element_size(), "mean_relative_error": rank_max(mean_relative.item()), "max_relative_error": rank_max(max_relative.item()), "collective": stats()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("minimal_minimax_m2_config.json"))
    parser.add_argument("--mode", choices=("baseline", "compressed", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--min-bytes", type=int, default=3 * 1024 * 1024)
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["NANOVLLM_FP8_ALL_REDUCE_MIN_BYTES"] = str(args.min_bytes)
    os.environ["NANOVLLM_FP8_ALL_REDUCE_STATS"] = "1"
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", rank)))
    config = PretrainedConfig.from_dict(json.loads(args.config.read_text()))
    torch.set_default_dtype(config.dtype)
    torch.set_default_device("cuda")
    model = MiniMaxM2ForCausalLM(config).eval()
    seed_weights(model, args.seed)
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print(json.dumps({"event": "metadata", "source_revision": source_revision(), "host": socket.gethostname(), "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()), "world_size": dist.get_world_size(), "seed": args.seed, "warmup": args.warmup, "iterations": args.iterations, "min_bytes": args.min_bytes, "cases": CASES}, sort_keys=True), flush=True)
    if not args.skip_validation:
        validation = validate_collective(args)
        if rank == 0:
            print(json.dumps({"event": "collective_validation", **validation}, sort_keys=True), flush=True)
    modes = ("baseline", "compressed") if args.mode == "both" else (args.mode,)
    with torch.inference_mode():
        for mode in modes:
            os.environ["NANOVLLM_FP8_ALL_REDUCE"] = "1" if mode == "compressed" else "0"
            for phase, batch, context in CASES:
                result = case(model, config, phase, batch, context, args)
                if rank == 0:
                    print(json.dumps({"event": "result", "mode": mode, **result}, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
