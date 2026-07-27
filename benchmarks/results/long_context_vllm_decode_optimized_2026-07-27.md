# Decode-optimized vLLM long-context A/B

This is the winning decode policy derived from the paper's latency lesson: keep small decode collectives on the native path and use compression only for large payloads. The paper's LL/sentinel/symmetric-memory NCCL kernels target NVLink-scale-up systems; this machine is PCIe SM120 with PyNCCL, so we did not claim or force an unsupported NCCL kernel port. Instead, the vLLM entry point now bypasses the FP8 helper entirely for sub-3 MiB tensors.

| input / output | stock prefill tok/s | optimized prefill tok/s | stock decode tok/s | optimized decode tok/s | stock wall | optimized wall | stock E2E generated tok/s | optimized E2E generated tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 / 512 | 4,803.82 | **7,311.48 (+52.20%)** | 21.73 | **21.92 (+0.85%)** | 26,050.2 ms | **25,176.2 ms (-3.36%)** | 19.65 | **20.34 (+3.47%)** |
| 20,000 / 512 | 4,546.27 | **6,745.61 (+48.38%)** | 21.57 | **21.80 (+1.06%)** | 28,642.6 ms | **26,907.0 ms (-6.06%)** | 17.88 | **19.03 (+6.45%)** |

These are three measured iterations per context after one warmup. Decode is now slightly faster than stock in both cases, so the prefill improvement converts into a real end-to-end gain.

## What changed

- `VLLM_FP8_ALL_REDUCE=1` with a 3 MiB threshold remains enabled for large BF16 reductions.
- `vllm/distributed/parallel_state.py` now checks the cached feature flag and payload size at the all-reduce entry point; sub-threshold decode reductions call stock vLLM directly.
- Feature flags, threshold, and diagnostics are cached once per worker process; `VLLM_FP8_ALL_REDUCE_STATS=1` is reserved for diagnostics and is off in performance runs.
- `VLLM_SHARDED_DOWNSTREAM=1` is not part of the winning long-context policy. Its isolated run added roughly 1–2% decode latency and did not improve prefill here.
- Native sequence-parallel/GEMM-communication overlap remains separately tested and fallback-gated because repeated large-shape requests hung on this PCIe topology.

Raw logs: `long_context_vllm_stock_3iter.log` and `long_context_vllm_fp8_inline_3iter.log`.
