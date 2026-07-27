import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import PretrainedConfig

from nanovllm.layers.embed_head import VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import LinearBase, NVFP4GroupedLinear, NVFP4LinearBase
from nanovllm.models.minimax_m2 import MiniMaxM2ForCausalLM
from nanovllm.utils.context import reset_context, set_context


def initialize_noise(model, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (NVFP4LinearBase, NVFP4GroupedLinear)):
                module.weight.view(torch.uint8).random_(0, 256)
                module.weight_scale_inv.fill_(1)
                module.weight_global_scale.fill_(
                    1.0 / (4 * math.sqrt(module.weight.shape[1] * 2))
                )
            elif isinstance(module, LinearBase):
                module.weight.normal_(0.0, 1.0 / math.sqrt(module.weight.shape[1]))
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, VocabParallelEmbedding):
                module.weight.normal_(0.0, 1.0 / math.sqrt(module.weight.shape[1]))
            elif isinstance(module, RMSNorm):
                module.weight.fill_(1.0)


def forward(model, input_ids, positions, use_fp8):
    os.environ["NANOVLLM_FP8_ALL_REDUCE"] = "1" if use_fp8 else "0"
    tokens = input_ids.numel()
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    set_context(
        True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=tokens,
        max_seqlen_k=tokens,
    )
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = model(input_ids, positions)
    end.record()
    end.synchronize()
    reset_context()
    return output, start.elapsed_time(end)


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    with Path("benchmarks/minimax_m2_dummy_config.json").open() as config_file:
        config = PretrainedConfig.from_dict(json.load(config_file))
    model = MiniMaxM2ForCausalLM(config)
    initialize_noise(model, 1234)
    model.eval()
    input_ids = torch.arange(512, dtype=torch.int64, device="cuda")
    positions = torch.arange(512, dtype=torch.int64, device="cuda")
    with torch.inference_mode():
        forward(model, input_ids, positions, False)
        forward(model, input_ids, positions, True)
        reference, reference_ms = forward(model, input_ids, positions, False)
        output, output_ms = forward(model, input_ids, positions, True)
        repeated, _ = forward(model, input_ids, positions, True)
    error = (output.float() - reference.float()).abs()
    mean_error = error.mean() / reference.float().abs().mean().clamp_min(1e-12)
    max_error = error.max() / reference.float().abs().max().clamp_min(1e-12)
    repeat_error = (repeated.float() - output.float()).abs().max()
    assert torch.isfinite(reference).all()
    assert torch.isfinite(output).all()
    assert repeat_error == 0
    if rank == 0:
        print(
            {
                "seed": 1234,
                "tokens": 512,
                "reference_ms": reference_ms,
                "fp8_ms": output_ms,
                "mean_error": mean_error.item(),
                "max_error": max_error.item(),
                "repeat_error": repeat_error.item(),
            }
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
