The NCCL ctypes integration in `nanovllm/layers/pynccl.py`, the specialized
router GEMM, and the fused Q/K RMSNorm protocol in `csrc/router_gemm.cu` are
adapted from vLLM commit
`1240c74c0a47473449cf0c3a9c2d87a1e159f73b`, Copyright vLLM contributors,
licensed under the Apache License 2.0.
