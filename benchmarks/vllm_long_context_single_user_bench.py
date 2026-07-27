"""End-to-end long-context concurrency benchmark for pinned official vLLM."""

from __future__ import annotations

import argparse
import json
import os
import time
import types

import torch
import torch.distributed as dist
from vllm import LLM, SamplingParams, TokensPrompt


class CaptureExtension:
    def capture_start(self):
        from vllm.distributed.compressed_all_reduce import reset_stats

        reset_stats()
        model = self.model_runner.model
        self.capture_events = []
        self.capture_forward = model.forward
        self.capture_logits = model.compute_logits

        def forward(_, *args, **kwargs):
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = self.capture_forward(*args, **kwargs)
            end.record()
            self.capture_events.append(("forward", input_ids.numel(), start, end))
            return output

        def compute_logits(_, hidden_states):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = self.capture_logits(hidden_states)
            end.record()
            self.capture_events.append(("logits", hidden_states.shape[0], start, end))
            return output

        model.forward = types.MethodType(forward, model)
        model.compute_logits = types.MethodType(compute_logits, model)

    def capture_stop(self):
        from vllm.distributed.compressed_all_reduce import get_stats

        torch.cuda.synchronize()
        events = [
            {"kind": kind, "tokens": tokens, "latency_ms": start.elapsed_time(end)}
            for kind, tokens, start, end in self.capture_events
        ]
        model = self.model_runner.model
        model.forward = self.capture_forward
        model.compute_logits = self.capture_logits
        del self.capture_events
        return {
            "rank": dist.get_rank(),
            "events": events,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "device": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "compressed_all_reduce": get_stats(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-tokens", type=int, nargs="+", default=[10000, 20000])
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--dummy-seed", type=int, default=1234)
    return parser.parse_args()


def sampling(output_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=output_tokens,
        detokenize=False,
    )


def make_prompt(prompt_tokens: int, salt: int) -> list[int]:
    start = (salt * 7919 + prompt_tokens) % 200000
    return [(start + index) % 200000 for index in range(prompt_tokens)]


def generate(
    llm: LLM,
    prompt_tokens: int,
    output_tokens: int,
    salt: int,
    concurrency: int,
) -> None:
    prompts = [
        TokensPrompt(prompt_token_ids=make_prompt(prompt_tokens, salt + request_index))
        for request_index in range(concurrency)
    ]
    llm.generate(prompts, sampling(output_tokens), use_tqdm=False)


def classify_model_events(
    rank_results: list[dict], iterations: int, decode_tokens: int
) -> tuple[float, float]:
    # Concurrent decode produces one forward/logits pair with one token per
    # active request. Prefill chunks are larger than that aggregate count.
    rank_totals = []
    for rank_result in rank_results:
        events = rank_result["events"]
        forwards = [event for event in events if event["kind"] == "forward"]
        logits = [event for event in events if event["kind"] == "logits"]
        if len(forwards) != len(logits):
            raise RuntimeError("forward/logits event count mismatch")
        prefill_ms = sum(
            forward["latency_ms"] + logit["latency_ms"]
            for forward, logit in zip(forwards, logits)
            if forward["tokens"] != decode_tokens
        ) / iterations
        decode_ms = sum(
            forward["latency_ms"] + logit["latency_ms"]
            for forward, logit in zip(forwards, logits)
            if forward["tokens"] == decode_tokens
        ) / iterations
        rank_totals.append((prefill_ms, decode_ms))
    return max(item[0] for item in rank_totals), max(item[1] for item in rank_totals)


def main() -> None:
    args = parse_args()
    if any(value < 1 for value in args.input_tokens):
        raise ValueError("input token counts must be positive")
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    if args.output_tokens < 1 or args.warmup < 0 or args.iterations < 1:
        raise ValueError("output tokens, warmup, and iterations must be positive")
    if args.max_model_len < max(args.input_tokens) + args.output_tokens:
        raise ValueError("max-model-len must cover prompt plus generated tokens")

    tuned = os.environ.get("VLLM_SHARDED_DOWNSTREAM", "0") == "1"
    compilation_config = {
        "mode": 3,
        "cudagraph_mode": "NONE",
        "pass_config": {"enable_qk_norm_rope_fusion": True},
    }
    if tuned:
        # This is vLLM's native allreduce+RMSNorm boundary.  It leaves the
        # FlashInfer-CUTLASS fused MoE routing path unchanged.
        compilation_config["pass_config"]["fuse_allreduce_rms"] = True
    llm = LLM(
        model=args.model,
        load_format="dummy",
        skip_tokenizer_init=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.concurrency,
        gpu_memory_utilization=0.7,
        kv_cache_dtype="fp8_e4m3",
        compilation_config=compilation_config,
        enable_prefix_caching=False,
        async_scheduling=False,
        disable_custom_all_reduce=True,
        worker_extension_cls="vllm_long_context_single_user_bench.CaptureExtension",
        seed=0,
    )
    print(
        json.dumps(
            {
                "event": "metadata",
                "engine": "vllm-pinned-stock",
                "model": args.model,
                "dummy_seed": args.dummy_seed,
                "input_tokens": args.input_tokens,
                "output_tokens": args.output_tokens,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "concurrency": args.concurrency,
                "tensor_parallel_size": args.tensor_parallel_size,
                "fp8_all_reduce": os.environ.get("VLLM_FP8_ALL_REDUCE", "0"),
                "fp8_all_reduce_min_bytes": os.environ.get(
                    "VLLM_FP8_ALL_REDUCE_MIN_BYTES", str(3 * 1024 * 1024)
                ),
                "sharded_downstream": tuned,
                "compilation_config": compilation_config,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for prompt_index, prompt_tokens in enumerate(args.input_tokens):
        for warmup_index in range(args.warmup):
            generate(
                llm,
                prompt_tokens,
                args.output_tokens,
                salt=100 + prompt_index * 10 + warmup_index,
                concurrency=args.concurrency,
            )
        wall_times = []
        phase_times = []
        last_capture_results = None
        for iteration in range(args.iterations):
            llm.llm_engine.collective_rpc("capture_start")
            start = time.perf_counter()
            generate(
                llm,
                prompt_tokens,
                args.output_tokens,
                salt=1000 + prompt_index * 100 + iteration,
                concurrency=args.concurrency,
            )
            wall_times.append((time.perf_counter() - start) * 1000.0)
            last_capture_results = llm.llm_engine.collective_rpc("capture_stop")
            phase_times.append(
                classify_model_events(last_capture_results, 1, args.concurrency)
            )
        prefill_ms = max(item[0] for item in phase_times)
        decode_ms = max(item[1] for item in phase_times)
        total_ms = sum(wall_times) / len(wall_times)
        result = {
            "event": "result",
            "engine": "vllm-pinned-stock",
            "prompt_tokens": prompt_tokens,
            "concurrency": args.concurrency,
            "generated_tokens": args.output_tokens,
            "prefill_ms_model": prefill_ms,
            "decode_ms_model": decode_ms,
            "wall_ms": total_ms,
            "prefill_tokens_per_second": (
                args.concurrency * prompt_tokens * 1000.0 / prefill_ms
            ),
            "decode_tokens_per_second": (
                args.concurrency * args.output_tokens * 1000.0 / decode_ms
            ),
            "end_to_end_generated_tokens_per_second": (
                args.concurrency * args.output_tokens * 1000.0 / total_ms
            ),
            "end_to_end_all_tokens_per_second": (
                args.concurrency
                * (prompt_tokens + args.output_tokens)
                * 1000.0
                / total_ms
            ),
            "allocated_bytes_per_rank": [item["allocated_bytes"] for item in last_capture_results],
            "compressed_all_reduce_per_rank": [
                item["compressed_all_reduce"] for item in last_capture_results
            ],
        }
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
