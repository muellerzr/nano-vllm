"""End-to-end single-user long-context benchmark for the current NanoVLLM tree."""

from __future__ import annotations

import argparse
import json
import os
import time

from nanovllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-tokens", type=int, nargs="+", default=[10000, 20000])
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dummy-seed", type=int, default=1234)
    return parser.parse_args()


def sampling(output_tokens: int) -> SamplingParams:
    return SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=output_tokens)


def make_prompt(prompt_tokens: int, salt: int) -> list[int]:
    # Distinct deterministic prefixes prevent Nano's block-hash cache from
    # turning a later measurement into a cached-prefix workload.
    start = (salt * 7919 + prompt_tokens) % 200000
    return [(start + index) % 200000 for index in range(prompt_tokens)]


def generate_warmup(
    llm: LLM,
    prompt_tokens: int,
    output_tokens: int,
    salt: int,
) -> None:
    prompt = make_prompt(prompt_tokens, salt)
    llm.generate([prompt], sampling(output_tokens), use_tqdm=False)


def measure_once(
    llm: LLM,
    prompt_tokens: int,
    output_tokens: int,
    salt: int,
) -> dict:
    prompt = make_prompt(prompt_tokens, salt)
    llm.add_request(prompt, sampling(output_tokens))
    prefill_ms = 0.0
    decode_ms = 0.0
    prefill_tokens = 0
    decode_tokens = 0
    start = time.perf_counter()
    while not llm.is_finished():
        step_start = time.perf_counter()
        _, scheduled_tokens = llm.step()
        elapsed_ms = (time.perf_counter() - step_start) * 1000.0
        if scheduled_tokens > 0:
            prefill_tokens += scheduled_tokens
            prefill_ms += elapsed_ms
        else:
            decode_tokens += -scheduled_tokens
            decode_ms += elapsed_ms
    total_ms = (time.perf_counter() - start) * 1000.0
    return {
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "prompt_tokens": prompt_tokens,
        "decode_steps": decode_tokens,
        "generated_tokens": output_tokens,
        "scheduled_prefill_tokens": prefill_tokens,
        "prefill_tokens_per_second": prompt_tokens * 1000.0 / prefill_ms,
        "decode_tokens_per_second": decode_tokens * 1000.0 / decode_ms,
        "end_to_end_generated_tokens_per_second": output_tokens * 1000.0 / total_ms,
        "end_to_end_all_tokens_per_second": (prompt_tokens + output_tokens) * 1000.0 / total_ms,
    }


def main() -> None:
    args = parse_args()
    if any(value < 1 for value in args.input_tokens):
        raise ValueError("input token counts must be positive")
    if args.output_tokens < 1 or args.warmup < 0 or args.iterations < 1:
        raise ValueError("output tokens, warmup, and iterations must be positive")
    if args.max_model_len < max(args.input_tokens) + args.output_tokens:
        raise ValueError("max-model-len must cover prompt plus generated tokens")

    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        load_format="dummy",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=0.7,
    )
    print(
        json.dumps(
            {
                "event": "metadata",
                "engine": "nanovllm-current",
                "model": args.model,
                "dummy_seed": args.dummy_seed,
                "input_tokens": args.input_tokens,
                "output_tokens": args.output_tokens,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "max_num_seqs": args.max_num_seqs,
                "tensor_parallel_size": args.tensor_parallel_size,
                "fp8_all_reduce": os.environ.get("NANOVLLM_FP8_ALL_REDUCE", "1"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for prompt_index, prompt_tokens in enumerate(args.input_tokens):
        for warmup_index in range(args.warmup):
            generate_warmup(
                llm,
                prompt_tokens,
                args.output_tokens,
                salt=100 + prompt_index * 10 + warmup_index,
            )
        measurements = [
            measure_once(
                llm,
                prompt_tokens,
                args.output_tokens,
                salt=1000 + prompt_index * 100 + iteration,
            )
            for iteration in range(args.iterations)
        ]
        result = {
            key: sum(item[key] for item in measurements) / len(measurements)
            for key in measurements[0]
            if key
            not in {
                "prompt_tokens",
                "generated_tokens",
                "decode_steps",
                "scheduled_prefill_tokens",
            }
        }
        result.update(
            {
                "event": "result",
                "engine": "nanovllm-current",
                "prompt_tokens": prompt_tokens,
                "generated_tokens": round(sum(item["generated_tokens"] for item in measurements) / len(measurements)),
                "decode_steps": round(sum(item["decode_steps"] for item in measurements) / len(measurements)),
                "scheduled_prefill_tokens": round(sum(item["scheduled_prefill_tokens"] for item in measurements) / len(measurements)),
            }
        )
        print(json.dumps(result, sort_keys=True), flush=True)
    llm.exit()


if __name__ == "__main__":
    main()
