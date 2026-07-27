# vLLM long-context A/B: stock versus adjusted vLLM

Both runs use the pinned nightly overlay, TP4, deterministic dummy seed 1234, one warmup, one measured iteration, compile mode 3, CUDA graphs disabled, FP8 E4M3 KV cache, FlashInfer attention, and FlashInfer-CUTLASS MoE. Prompt salts are unique and deterministic so neither run reuses a cached 10k prefix. The adjusted run enables FP8 compressed TP all-reduce at the 3 MiB threshold and vLLM's native `fuse_allreduce_rms` downstream boundary. Native SP overlap is intentionally off for these repeated long-context requests because the earlier SM120 PCIe tests hung; its isolated results are recorded in `experimental_mechanisms_2026-07-26.md`.

| input / output | stock prefill tok/s | adjusted prefill tok/s | stock decode tok/s | adjusted decode tok/s | stock end-to-end generated tok/s | adjusted end-to-end generated tok/s | adjusted wall delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 / 512 | 4,812.88 | **7,343.36** (+52.58%) | 22.09 | 20.66 (-6.48%) | **19.88** | 19.21 (-3.37%) | +3.49% |
| 20,000 / 512 | 4,555.45 | **6,762.19** (+48.44%) | 21.99 | 20.44 (-7.06%) | **18.17** | 17.95 (-1.21%) | +1.23% |

Practical result: the adjustments materially improve prefill, but do not yet improve single-user end-to-end generation at these output lengths. Decode dominates the wall time, and the tuned path is 1.2–3.4% slower end-to-end in this one-warmup/one-iteration measurement. The next optimization target is decode; the prefill result is already a clear win.

The adjusted run selected 625 compressed collectives at 10k and 1,250 at 20k. Most small reductions correctly fell back: 63,875 threshold fallbacks in each run. Per-rank allocated memory was identical at 71,725,642,752 bytes. The FP8 and downstream mechanisms were correctness-checked separately in the isolated matrix; see `experimental_mechanisms_2026-07-26.md`.

Environment and exact values are recorded in `long_context_vllm_ab_2026-07-26.json`. Raw logs are `long_context_vllm_stock_final.log` and `long_context_vllm_tuned_final.log`.
