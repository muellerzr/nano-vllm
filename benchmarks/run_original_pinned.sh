#!/usr/bin/env bash
set -euo pipefail

docker run --rm --gpus all --ipc=host \
  --entrypoint /bin/bash \
  -e BENCH_IMAGE=vllm/vllm-openai:nightly-x86_64 \
  -v /home/zach/vllm-minimax-dummy:/old:ro \
  -v /home/zach/vllm-minimax-dummy/published:/model:ro \
  vllm/vllm-openai@sha256:929e0ce173d6c2b44adabb6349ad6988710ca43ad9fd7fa82869cc55857f201d \
  -lc \
  "/usr/bin/python3 /old/vllm_minimax_m2_dummy_bench.py --model /model --warmup 5 --iterations 100 --kv-cache-dtype auto --compilation-config '{\"mode\":3,\"cudagraph_mode\":\"NONE\",\"pass_config\":{\"enable_qk_norm_rope_fusion\":true}}'"
