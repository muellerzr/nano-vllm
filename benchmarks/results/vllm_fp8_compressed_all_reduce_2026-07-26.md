# vLLM FP8 compressed all-reduce investigation

The first mechanism is now implemented and validated in isolation behind `VLLM_FP8_ALL_REDUCE`. The deployment image starts from the pinned nightly digest and retains its compiled extensions; the overlay only adds the policy/collective module and a small `parallel_state.py` hook. The feature-off path returns to stock vLLM behavior. `VLLM_FP8_ALL_REDUCE_MIN_BYTES` defaults to 3 MiB.

## What changed in the benchmark

The pinned nightly currently emits chunked prefill events (for example, 16 followed by 496 tokens) even with the published configuration. The original parser treated each chunk as a complete request and therefore could not reproduce the four logical cases. The reviewed harness now groups scheduler chunks until the requested logical prompt token count is reached, then sums the forward-plus-logits CUDA-event times. Decode groups retain the prompt followed by the generated-token event. This is a measurement adaptation to current vLLM scheduling; it does not change model inputs or the requested shapes.

The resulting feature-off control is 153.234840, 45.310126, 418.156781, and 45.768664 ms for B32/C16 prefill, B1/C16 decode, B4/C512 prefill, and B1/C2048 decode. These are not directly comparable to the older 109.365314 ms B32/C16 published artifact because that artifact was measured before the current scheduler emitted chunked events.

## Native vLLM audit

The pinned runtime already contains `fuse_gemm_comms` and sequence-parallel MoE support. Sequence-parallel MoE requires expert parallelism plus data parallelism greater than one, so it is not active in this TP4/DP1 MiniMax configuration. On this four-PCIe-SM120 host, PyNCCL is selected; NCCL symmetric-memory all-reduce reports SM120 unsupported, and vLLM custom all-reduce reports more than two PCIe-only GPUs unsupported.

## Collective gate

The candidate accepts only contiguous CUDA BF16 sum reductions whose BF16 payload is at least 3 MiB, whose element count is divisible by 128, whose device capability is at least SM90, and for which PyNCCL is available. It performs a per-128-element global-maximum scale reduction, FP8 E4M3 quantization, FP8 sum, and BF16 dequantization. Unsupported dtype, device, layout, size, capability, backend, or disabled flag falls back to the stock collective.

The isolated TP4 test selected the candidate at and above the inclusive 3 MiB threshold:

| tokens | BF16 bytes | selected | stock ms | FP8 ms | speedup | mean rel. error | max rel. error |
|---:|---:|:---:|---:|---:|---:|---:|---:|
| 1 | 6,144 | no | 0.026405 | 0.032438 | 0.814× | 0 | 0 |
| 16 | 98,304 | no | 0.038906 | 0.038873 | 1.001× | 0 | 0 |
| 128 | 786,432 | no | 0.186417 | 0.185842 | 1.003× | 0 | 0 |
| 512 | 3,145,728 | yes | 0.743986 | 0.459631 | 1.619× | 4.57% | 9.45% |
| 2,048 | 12,582,912 | yes | 2.890182 | 1.632683 | 1.770× | 4.57% | 10.39% |

## Model correctness gate

With deterministic dummy weights/inputs (seed 1234), feature-off and feature-on were run for one iteration over all four shapes. The first 256 flattened logits were identical for the threshold-fallback cases. Compressed cases differed by at most `1.0133e-6` absolute (`2.3644e-7` mean absolute) in the captured sample. This is a smoke check of model output; the collective error bounds above remain the governing numerical characterization.

## Four-shape A/B

Both runs used TP4, 5 warmups, 100 measured iterations, CUDA graphs disabled, compile mode 3, QK/norm/rope fusion enabled, deterministic scheduler seed 0, and the same wheel-overlay image.

| shape | feature off (ms) | feature on (ms) | speedup | selection |
|---|---:|---:|---:|---|
| prefill B32/C16 | 153.234840 | 154.817143 | 0.990× | threshold fallback |
| decode B1/C16 | 45.310126 | 45.807804 | 0.989× | threshold fallback |
| prefill B4/C512 | 418.156781 | 275.712669 | 1.517× | 25,000 compressed calls/rank |
| decode B1/C2048 | 45.768664 | 46.287477 | 0.989× | 12,500 compressed + 12,500 fallback calls/rank |

Allocated memory was 35,898,358,784 bytes/rank with the feature off and 35,899,374,592 bytes/rank with it on. All four ranks reported RTX PRO 6000 Blackwell Max-Q, compute capability 12.0. The stopped `glm-ray-worker` container was not modified.

Nineteen dependency-free local tests cover the policy boundaries, inclusive threshold, payload accounting, single-forward grouping, chunked-prefill grouping, decode grouping, incomplete-request rejection, native-SP isolated-shape selection, and native allreduce+RMSNorm selection. `git diff --check` and Python bytecode compilation also pass.

The compressed collective, native allreduce+RMSNorm downstream fusion, and the stable isolated native-SP cases are now independently selected behind separate flags. The native-SP policy keeps the two known hanging shapes on stock vLLM. See `experimental_mechanisms_2026-07-26.md` for the consolidated matrix.
