# MiniMax M2 experimental mechanism matrix

All rows use the pinned nightly image/source, TP4, deterministic dummy seed 1234, scheduler seed 0, compile mode 3, CUDA graphs disabled, 5 warmups, 100 iterations, and CUDA-event timing around model forward plus logits with rank-max aggregation. The deployed attention and MoE backends remain FlashInfer and FlashInfer-CUTLASS.

Control: **153.234840 / 45.310126 / 418.156781 / 45.768664 ms** for B32/C16 prefill, B1/C16 decode, B4/C512 prefill, and B1/C2048 decode.

| mechanism | B32/C16 | B1/C16 | B4/C512 | B1/C2048 | result |
|---|---:|---:|---:|---:|---|
| control | 153.235 ms | 45.310 ms | 418.157 ms | 45.769 ms | stock vLLM/PyNCCL |
| FP8 compressed all-reduce | 154.817 ms | 45.808 ms | **275.713 ms** | 46.287 ms | **1.517×** at 3 MiB+; other shapes fallback |
| native overlap (selected isolated cases) | 380.386 ms | 87.244 ms | 418.157 ms (stock) | 45.769 ms (stock) | vLLM native SP selected only for stable cases; large-shape SP repeat hang avoided |
| native allreduce+RMSNorm downstream | 151.425 ms | 44.729 ms | 418.510 ms | 45.461 ms | selected `fuse_allreduce_rms`; fused MoE unchanged |

The FP8 collective’s isolated communication gain is 1.619× at exactly 3 MiB and 1.770× at 12 MiB, with 4.57% mean relative error and 9.45–10.39% maximum relative error. The deterministic model smoke comparison differed by at most `1.0133e-6` absolute in the captured logits sample.

The native-overlap path uses vLLM’s own `enable_sp` + `fuse_gemm_comms`, with `sp_min_token_num=1` and the benchmark’s decode grouping adapted to the local `batch*4` token count. Its full 5/100 A/B completed for B32/C16 and B1/C16, but repeated B4/C512 prefill and B1/C2048 decode requests exhausted vLLM shared-memory scheduling and never emitted a result, even after disabling chunked prefill and raising `max_num_batched_tokens` to 4096. The benchmark now requires isolated per-shape execution for native SP and keeps those two shapes on stock vLLM (`shape_repeat_unsupported`) instead of hanging. Captured 256-logit samples were exactly equal for both selected full cases; the selected-path smoke memory was 35,931,915,264 bytes/rank.

The downstream mechanism is adapted to the native vLLM `fuse_allreduce_rms` boundary. This keeps the FlashInfer-CUTLASS fused MoE routing implementation intact and avoids premature full-hidden reconstruction. All four full A/B cases selected the native fusion, matched the stock captured 256-logit sample exactly, and used 35,898,358,784 bytes/rank.

The capability/size/backend policy remains explicit: unsupported dtypes, hardware, payloads, backend availability, native-SP shape repetition, or fused-MoE constraints retain stock vLLM behavior. The isolated-case support is implemented in `benchmarks/vllm_minimax_m2_dummy_bench.py` and exposed through `VLLM_BENCH_CASE` in `benchmarks/run_vllm_minimax_m2.sh`.
