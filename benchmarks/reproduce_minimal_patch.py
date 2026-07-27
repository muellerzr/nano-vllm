#!/usr/bin/env python3
"""Reproduce stock vs FP8-collective NanoVLLM timings with synthetic weights."""

import argparse
import gc
import json
import math
import os
import platform
import socket
import subprocess
from time import perf_counter
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import PretrainedConfig

from nanovllm.layers.compressed_collective import all_reduce, reset_stats, stats
from nanovllm.models.minimax_m2 import MiniMaxM2ForCausalLM
from nanovllm.sampling_params import SamplingParams
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
    collective = stats()
    for key in ("compressed_calls", "fallback_calls", "input_bytes", "payload_bytes"):
        collective[key] //= iterations
    return {
        "latency_ms": latency_ms,
        "tokens_per_second": tokens * 1000 / latency_ms,
        "collective": collective,
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
    parser.add_argument("--profile", choices=("forward", "long-context"), default="forward")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("minimal_minimax_m2_config.json"))
    parser.add_argument("--mode", choices=("baseline", "compressed", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--min-bytes", type=int, default=3 * 1024 * 1024)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--input-tokens", type=int, default=10000)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    return parser.parse_args()


def make_prompt(length: int, vocab_size: int) -> list[int]:
    return [(17 * i + 3) % vocab_size for i in range(length)]


def measure_generation(llm, prompts, sampling_params, warmup: int, iterations: int, vocab_size: int) -> dict:
    def once(salt: int):
        # Prefix caching is disabled in the vLLM protocol.  Nano's block
        # manager hashes blocks, so every sample gets a distinct first block.
        run_prompts = [[(token + salt) % vocab_size for token in prompt] for prompt in prompts]
        for prompt in run_prompts:
            llm.add_request(prompt, sampling_params)
        prefill_seconds = decode_seconds = 0.0
        prefill_tokens = decode_tokens = 0
        start = perf_counter()
        while not llm.is_finished():
            step_start = perf_counter()
            _, scheduled = llm.step()
            elapsed = perf_counter() - step_start
            if scheduled > 0:
                prefill_tokens += scheduled
                prefill_seconds += elapsed
            else:
                decode_tokens += -scheduled
                decode_seconds += elapsed
        return perf_counter() - start, prefill_tokens, prefill_seconds, decode_tokens, decode_seconds

    for index in range(warmup):
        once(7919 + index)
    samples = [once(100000 + index) for index in range(iterations)]
    wall = sum(x[0] for x in samples) / iterations
    prefill_s = sum(x[2] for x in samples) / iterations
    decode_s = sum(x[4] for x in samples) / iterations
    prompt_tokens = len(prompts) * len(prompts[0])
    generated_tokens = len(prompts) * sampling_params.max_tokens
    return {
        "concurrency": len(prompts),
        "input_tokens": len(prompts[0]),
        "output_tokens": sampling_params.max_tokens,
        "wall_ms": wall * 1000,
        "prefill_ms": prefill_s * 1000,
        "decode_ms": decode_s * 1000,
        "prefill_tokens": prompt_tokens,
        "decode_tokens": generated_tokens,
        "prefill_tokens_per_second": prompt_tokens / prefill_s,
        "decode_tokens_per_second": generated_tokens / decode_s,
        "generated_tokens_per_second": generated_tokens / wall,
        "generated_tokens_per_second_per_user": generated_tokens / len(prompts) / wall,
    }


def run_long_context(args) -> None:
    from nanovllm import LLM

    config = json.loads(args.config.read_text())
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    os.environ["NANOVLLM_FP8_ALL_REDUCE_MIN_BYTES"] = str(args.min_bytes)
    os.environ["NANOVLLM_FP8_ALL_REDUCE_STATS"] = "1"
    prompts = [make_prompt(args.input_tokens, config["vocab_size"]) for _ in range(args.concurrency)]
    params = SamplingParams(temperature=1.0, max_tokens=args.output_tokens, ignore_eos=True)
    model_dir = args.config.parent / "minimal_model"
    modes = ("baseline", "compressed") if args.mode == "both" else (args.mode,)
    for mode in modes:
        os.environ["NANOVLLM_FP8_ALL_REDUCE"] = "1" if mode == "compressed" else "0"
        llm = LLM(
            str(model_dir),
            load_format="dummy",
            tensor_parallel_size=args.world_size or 4,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.concurrency,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=True,
        )
        result = measure_generation(llm, prompts, params, args.warmup, args.iterations, config["vocab_size"])
        result.update({"mode": mode, "source_revision": source_revision(), "seed": args.seed,
                       "world_size": args.world_size or 4, "warmup": args.warmup, "iterations": args.iterations,
                       "config": str(args.config), "model": str(model_dir)})
        print(json.dumps({"event": "long_context_result", **result}, sort_keys=True), flush=True)
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def run(args) -> None:
    if args.profile == "long-context":
        run_long_context(args)
        return
    os.environ["NANOVLLM_FP8_ALL_REDUCE_MIN_BYTES"] = str(args.min_bytes)
    os.environ["NANOVLLM_FP8_ALL_REDUCE_STATS"] = "1"
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
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


def _spawn_rank(rank: int, world_size: int, args) -> None:
    os.environ.update(
        MASTER_ADDR=os.getenv("MASTER_ADDR", "127.0.0.1"),
        MASTER_PORT=os.getenv("MASTER_PORT", "29501"),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    run(args)


def main() -> None:
    args = parse_args()
    if args.profile == "long-context":
        run(args)
        return
    if "RANK" not in os.environ:
        world_size = args.world_size or torch.cuda.device_count()
        if world_size < 2:
            raise RuntimeError("TP reproduction requires at least two CUDA devices")
        torch.multiprocessing.spawn(_spawn_rank, args=(world_size, args), nprocs=world_size)
    else:
        run(args)


if __name__ == "__main__":
    main()
