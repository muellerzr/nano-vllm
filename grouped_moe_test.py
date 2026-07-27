import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from nanovllm.layers.nvfp4 import grouped_nvfp4_mm, quant_nvfp4, quant_nvfp4_fixed
from nanovllm.models.minimax_m2 import MiniMaxM2SparseMoeBlock


def load_weights(linear):
    for expert in range(linear.weight.shape[0]):
        weight = torch.randn(
            linear.weight.shape[2],
            linear.weight.shape[1] * 2,
            device="cuda",
            dtype=torch.bfloat16,
        ) / (linear.weight.shape[1] * 2) ** 0.5
        packed, scales, global_scale = quant_nvfp4(weight)
        linear.weight.data[expert].copy_(packed.t())
        linear.weight_scale_inv.data[expert].copy_(scales)
        linear.weight_global_scale.data[expert].copy_(global_scale)


def apply(linear, x, expert):
    tokens = x.shape[0]
    padded_tokens = (tokens + 127) // 128 * 128
    padded = torch.zeros(
        padded_tokens,
        x.shape[1],
        dtype=x.dtype,
        device=x.device,
    )
    padded[:tokens] = x
    padded, scales, global_scale = quant_nvfp4_fixed(
        padded,
        linear.input_global_scale,
    )
    output = grouped_nvfp4_mm(
        padded.reshape(1, padded_tokens, x.shape[1] // 2),
        linear.weight,
        scales,
        linear.weight_scale_inv,
        global_scale,
        linear.weight_global_scale,
        expert.reshape(1).to(torch.int32),
        torch.tensor([tokens], dtype=torch.int32, device=x.device),
    )
    return output[0, :tokens]


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    config = SimpleNamespace(
        hidden_size=3072,
        intermediate_size=1536,
        num_experts_per_tok=2,
        num_local_experts=4,
    )
    torch.manual_seed(0)
    block = MiniMaxM2SparseMoeBlock(config)
    block.gate.weight.data.normal_()
    load_weights(block.w1_w3)
    load_weights(block.w2)
    hidden_states = torch.randn(32, 3072)
    router_logits = block.gate(hidden_states)
    routing_weights = torch.sigmoid(router_logits.float())
    scores = routing_weights + block.e_score_correction_bias
    top_k_index = torch.topk(scores, block.top_k, dim=-1, sorted=False).indices
    top_k_weights = routing_weights.gather(1, top_k_index)
    top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
    reference = torch.zeros_like(hidden_states)
    for expert in top_k_index.unique():
        token_idx, slot_idx = torch.where(top_k_index == expert)
        y = apply(block.w1_w3, hidden_states[token_idx], expert)
        y = block.act_fn(y)
        y = apply(block.w2, y, expert)
        reference.index_add_(
            0,
            token_idx,
            y * top_k_weights[token_idx, slot_idx, None].to(y.dtype),
        )
    dist.all_reduce(reference)
    output = block(hidden_states)
    error = (output.float() - reference.float()).abs()
    mean_error = error.mean() / reference.float().abs().mean().clamp_min(1e-12)
    max_error = error.max() / reference.float().abs().max().clamp_min(1e-12)
    assert torch.isfinite(output).all()
    assert mean_error < 0.001
    assert max_error < 0.001
    if rank == 0:
        print({"max_error": max_error.item(), "mean_error": mean_error.item()})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
