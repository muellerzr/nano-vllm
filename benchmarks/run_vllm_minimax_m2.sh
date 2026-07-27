#!/usr/bin/env bash
set -euo pipefail

exec /usr/bin/python3 /benchmarks/vllm_minimax_m2_dummy_bench.py \
  --model /model \
  --warmup "${WARMUP:-5}" \
  --iterations "${ITERATIONS:-100}" \
  --dummy-seed "${DUMMY_SEED:-1234}" \
  --scheduler-seed "${SCHEDULER_SEED:-0}" \
  ${DISABLE_CHUNKED_PREFILL:+--disable-chunked-prefill} \
  ${VLLM_NATIVE_OVERLAP:+--native-overlap} \
  ${VLLM_NATIVE_OVERLAP_FORCE:+--force-native-overlap} \
  ${VLLM_SP_MIN_TOKEN_NUM:+--sp-min-token-num "${VLLM_SP_MIN_TOKEN_NUM}"} \
  ${VLLM_SHARDED_DOWNSTREAM:+--sharded-downstream} \
  ${VLLM_BENCH_CASE:+--case "${VLLM_BENCH_CASE}"} \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}" \
  --compilation-config \
  '{"mode":3,"cudagraph_mode":"NONE","pass_config":{"enable_qk_norm_rope_fusion":true}}'
