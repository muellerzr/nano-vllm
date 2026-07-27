import argparse
import json
import os
import platform
import socket
import types

import torch
import torch.distributed as dist
from vllm import LLM, SamplingParams, TokensPrompt

from event_grouping import group_event_indices
from feature_policy import (
    native_overlap_case_decision,
    native_overlap_decision,
    sharded_downstream_decision,
)
from experiment_config import build_compilation_config


LABEL = "vLLM MiniMax M2 faithful dummy benchmark"


class CaptureExtension:

    def capture_start(self):
        from vllm.distributed.compressed_all_reduce import reset_stats

        reset_stats()
        model = self.model_runner.model
        self.capture_events = []
        self.capture_output_sample = None
        self.capture_output_stats = (
            os.environ.get("VLLM_CAPTURE_OUTPUT_SAMPLE", "0") == "1"
        )
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
            if self.capture_output_stats and self.capture_output_sample is None:
                self.capture_output_sample = (
                    output.flatten()[:256].detach().float().cpu().tolist()
                )
            self.capture_events.append(
                ("logits", hidden_states.shape[0], start, end)
            )
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
            "output_sample": self.capture_output_sample,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--compilation-config", default='{"mode": 0}')
    parser.add_argument("--dummy-seed", type=int, default=1234)
    parser.add_argument("--scheduler-seed", type=int, default=0)
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--capture-output-sample", action="store_true")
    parser.add_argument("--native-overlap", action="store_true")
    parser.add_argument("--force-native-overlap", action="store_true")
    parser.add_argument("--sp-min-token-num", type=int)
    parser.add_argument("--sharded-downstream", action="store_true")
    parser.add_argument(
        "--case",
        choices=("prefill_b32_c16", "decode_b1_c16", "prefill_b4_c512", "decode_b1_c2048"),
        help="Run one shape in an isolated process; useful for native SP paths that retain scheduler state.",
    )
    return parser.parse_args()


def rpc(llm, method):
    return llm.llm_engine.collective_rpc(method)


def generate(llm, batch, context, output_len):
    prompts = [
        TokensPrompt(prompt_token_ids=[0] * context)
        for _ in range(batch)
    ]
    sampling = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=output_len,
        detokenize=False,
    )
    llm.generate(prompts, sampling, use_tqdm=False)


def benchmark_case(
    llm,
    phase,
    batch,
    context,
    warmup,
    iterations,
    *,
    native_overlap_selected=False,
):
    output_len = 1 if phase == "prefill" else 2
    for _ in range(warmup):
        generate(llm, batch, context, output_len)

    rpc(llm, "capture_start")
    for _ in range(iterations):
        generate(llm, batch, context, output_len)
    rank_results = rpc(llm, "capture_stop")

    rank_latencies = []
    for rank_result in rank_results:
        forwards = [
            event for event in rank_result["events"] if event["kind"] == "forward"
        ]
        logits = [
            event for event in rank_result["events"] if event["kind"] == "logits"
        ]
        if len(forwards) != len(logits):
            raise RuntimeError("forward and logits event counts differ")
        groups = group_event_indices(
            [event["tokens"] for event in forwards],
            phase,
            batch=batch,
            context=context,
            iterations=iterations,
            decode_token_count=(batch * 4 if native_overlap_selected else None),
        )
        rank_latencies.append(
            sum(
                sum(
                    forwards[index]["latency_ms"] + logits[index]["latency_ms"]
                    for index in group
                )
                for group in groups
            )
            / iterations
        )

    latency_ms = max(rank_latencies)
    tokens = batch * context if phase == "prefill" else batch
    return {
        "event": "result",
        "phase": phase,
        "batch": batch,
        "context": context,
        "tokens_per_iteration": tokens,
        "latency_ms": latency_ms,
        "tokens_per_second": tokens * 1000.0 / latency_ms,
        "rank_latency_ms": rank_latencies,
        "compressed_all_reduce_per_rank": [
            item["compressed_all_reduce"] for item in rank_results
        ],
        "output_sample": rank_results[0]["output_sample"],
    }, rank_results


def main():
    args = parse_args()
    if args.capture_output_sample:
        os.environ["VLLM_CAPTURE_OUTPUT_SAMPLE"] = "1"
    with open(os.path.join(args.model, "config.json")) as config_file:
        model_config = json.load(config_file)
    capability = torch.cuda.get_device_capability()
    max_tokens = max(32 * 16, 4 * 512, 1 * 2048)
    base_overlap_selected, base_overlap_reason, overlap_threshold = native_overlap_decision(
        requested=args.native_overlap,
        force=args.force_native_overlap,
        capability_major=capability[0],
        hidden_size=int(model_config["hidden_size"]),
        tensor_parallel_size=4,
        tokens=max_tokens,
        element_size=2,
        backend_available=True,
    )
    case_names = {
        "prefill_b32_c16": ("prefill", 32, 16),
        "decode_b1_c16": ("decode", 1, 16),
        "prefill_b4_c512": ("prefill", 4, 512),
        "decode_b1_c2048": ("decode", 1, 2048),
    }
    if args.case is None:
        overlap_selected, overlap_reason = native_overlap_case_decision(
            selected=base_overlap_selected,
            isolated_case=False,
            phase="",
            batch=0,
            context=0,
        )
    else:
        case_phase, case_batch, case_context = case_names[args.case]
        overlap_selected, overlap_reason = native_overlap_case_decision(
            selected=base_overlap_selected,
            isolated_case=True,
            phase=case_phase,
            batch=case_batch,
            context=case_context,
        )
    sharded_selected, sharded_reason = sharded_downstream_decision(
        requested=args.sharded_downstream,
        fused_moe_requires_full_hidden=True,
        backend_available=True,
        native_allreduce_rms_available=(capability[0] >= 9),
    )
    compilation_config = build_compilation_config(
        json.loads(args.compilation_config),
        overlap=overlap_selected,
        sharded_downstream=sharded_selected,
        sp_min_token_num=(
            args.sp_min_token_num
            if args.sp_min_token_num is not None
            else (1 if args.force_native_overlap else None)
        ),
    )
    max_model_len = 4096
    max_num_batched_tokens = 2048 if not args.disable_chunked_prefill else max_model_len
    llm = LLM(
        model=args.model,
        load_format="dummy",
        skip_tokenizer_init=True,
        tensor_parallel_size=4,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=32,
        gpu_memory_utilization=0.7,
        kv_cache_memory_bytes=1 << 30,
        kv_cache_dtype=args.kv_cache_dtype,
        enforce_eager=compilation_config.get("mode", 0) == 0,
        compilation_config=compilation_config,
        enable_prefix_caching=False,
        async_scheduling=False,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        disable_custom_all_reduce=True,
        worker_extension_cls=(
            "vllm_minimax_m2_dummy_bench.CaptureExtension"
        ),
        seed=args.scheduler_seed,
    )
    print(
        json.dumps(
            {
                "event": "metadata",
                "label": LABEL,
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "vllm": __import__("vllm").__version__,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "image": os.environ.get("BENCH_IMAGE"),
                "model": args.model,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "dummy_seed": args.dummy_seed,
                "scheduler_seed": args.scheduler_seed,
                "disable_chunked_prefill": args.disable_chunked_prefill,
                "max_num_batched_tokens": max_num_batched_tokens,
                "kv_cache_dtype": args.kv_cache_dtype,
                "compilation_config": compilation_config,
                "native_overlap": args.native_overlap,
                "force_native_overlap": args.force_native_overlap,
                "native_overlap_selected": overlap_selected,
                "native_overlap_reason": overlap_reason,
                "native_overlap_base_selected": base_overlap_selected,
                "native_overlap_base_reason": base_overlap_reason,
                "native_overlap_token_threshold": overlap_threshold,
                "sharded_downstream": args.sharded_downstream,
                "sharded_downstream_selected": sharded_selected,
                "sharded_downstream_reason": sharded_reason,
                "fp8_all_reduce": os.environ.get("VLLM_FP8_ALL_REDUCE", "0"),
                "fp8_all_reduce_min_bytes": int(
                    os.environ.get(
                        "VLLM_FP8_ALL_REDUCE_MIN_BYTES",
                        str(3 * 1024 * 1024),
                    )
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    all_rank_results = None
    cases = [
        ("prefill", 32, 16),
        ("decode", 1, 16),
        ("prefill", 4, 512),
        ("decode", 1, 2048),
    ]
    if args.case is not None:
        cases = [case_names[args.case]]
    for phase, batch, context in cases:
        result, all_rank_results = benchmark_case(
            llm,
            phase,
            batch,
            context,
            args.warmup,
            args.iterations,
            native_overlap_selected=overlap_selected,
        )
        print(json.dumps(result, sort_keys=True), flush=True)

    print(
        json.dumps(
            {
                "event": "memory",
                "allocated_bytes_per_rank": [
                    result["allocated_bytes"] for result in all_rank_results
                ],
                "devices": [result["device"] for result in all_rank_results],
                "compute_capabilities": [
                    result["compute_capability"] for result in all_rank_results
                ],
                "case": args.case,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
