# 10k-context vLLM concurrency scaling

These runs are paired stock-versus-optimized vLLM measurements: 10,000 input tokens and 512 generated tokens per request, TP4, one warmup and one measurement in a fresh container per point. Aggregate throughput counts all concurrent requests. The optimized policy is FP8 compressed all-reduce above 3 MiB with the sub-threshold inline native-all-reduce bypass; native downstream fusion remains off for this workload.

| concurrency | stock prefill tok/s | optimized prefill tok/s | stock decode tok/s | optimized decode tok/s | stock E2E generated tok/s | optimized E2E generated tok/s | E2E gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4,399.6 | **6,432.6** (+46.21%) | 89.40 | 88.86 (-0.60%) | 62.89 | **68.67** | **+9.19%** |
| 8 | 4,322.0 | **6,277.0** (+45.23%) | 186.28 | 185.96 (-0.17%) | 99.65 | **115.78** | **+16.20%** |
| 16 | 4,277.6 | **6,196.5** (+44.86%) | 392.36 | **397.99 (+1.43%)** | 138.77 | **173.86** | **+25.28%** |
| 32 | 4,225.9 | **6,103.5** (+44.43%) | 948.37 | 943.37 (-0.53%) | 174.32 | **231.39** | **+32.74%** |

All four requested concurrency levels fit the available KV-cache budget and completed without OOM or scheduler failure. Raw logs are `long_context_vllm_stock_c4.log`, `long_context_vllm_opt_c4.log`, and the corresponding c8/c16/c32 files.
