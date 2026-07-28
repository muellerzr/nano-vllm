# NanoVLLM MiniMax M2

A minimal fixed-target inference engine for MiniMax M2 on four SM120 GPUs.

The runtime uses tensor parallelism across four GPUs, BF16 attention,
FlashInfer paged attention and CUTLASS NVFP4 MoE, FP8 E4M3 KV cache, and
FP8-compressed BF16 all-reduce above a 3 MiB threshold. It has no vLLM Python
or package dependency.

The deployment environment must provide PyTorch with CUDA, Triton, FlashInfer,
NCCL, Transformers, safetensors, and the CUDA toolkit.

```bash
TORCH_CUDA_ARCH_LIST=12.0 pip install --no-build-isolation --no-deps .
```

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/model",
    tensor_parallel_size=4,
    max_model_len=32768,
    max_num_batched_tokens=2048,
)
outputs = llm.generate(
    [[1, 2, 3]],
    SamplingParams(temperature=1.0, max_tokens=512),
)
```

Synthetic correctness, forward, and long-context reproduction are all driven
by one script:

```bash
python benchmarks/reproduce_minimal_patch.py --profile forward --mode both
python benchmarks/reproduce_minimal_patch.py \
  --profile long-context \
  --mode both \
  --input-tokens 10000 \
  --output-tokens 512 \
  --concurrency 1
```

The reproduction uses deterministic synthetic weights with seed 1234 and does
not download model weights.
