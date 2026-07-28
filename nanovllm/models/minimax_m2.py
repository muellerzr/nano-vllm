import torch
from torch import nn
import torch.distributed as dist
import flashinfer

from nanovllm.layers.attention import Attention
from nanovllm.layers.compressed_collective import all_reduce
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import NVFP4GroupedLinear, QKVParallelLinear, RouterLinear, RowParallelLinear
from nanovllm.layers.minimax_rms_norm import qk_rms_norm
from nanovllm.layers.router import topk_sigmoid
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class MiniMaxM2Attention(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = config.num_key_value_heads // tp_size
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.q_norm = RMSNorm(self.q_size, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.kv_size, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=config.rotary_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.attn = Attention()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k = qk_rms_norm(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            self.q_size,
            self.kv_size,
            self.q_norm.eps,
        )
        _, _, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        cache = self.rotary_emb.cos_sin_cache
        if cache.ndim == 3:
            cache = cache[:, 0]
        flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(
            positions,
            q,
            k,
            self.head_dim,
            cache,
            True,
        )
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        o = self.attn(q, k, v)
        return self.o_proj(o.flatten(1, -1))


class MiniMaxM2SparseMoeBlock(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.gate = RouterLinear(config.hidden_size, self.num_experts)
        tp_size = dist.get_world_size()
        self.tp_size = tp_size
        self.tp_rank = dist.get_rank()
        self.w1_w3 = NVFP4GroupedLinear(self.num_experts, config.hidden_size, config.intermediate_size * 2 // tp_size)
        self.w2 = NVFP4GroupedLinear(self.num_experts, config.intermediate_size // tp_size, config.hidden_size)
        self._flashinfer_moe = flashinfer.fused_moe.cutlass_fused_moe
        self._flashinfer_quant_scales = [
            self.w1_w3.input_global_scale.expand(self.num_experts),
            self.w1_w3.weight_scale_inv.view(torch.int32),
            self.w1_w3.weight_global_scale,
            self.w2.input_global_scale.expand(self.num_experts),
            self.w2.weight_scale_inv.view(torch.int32),
            self.w2.weight_global_scale,
        ]
        self._w1_packed = self.w1_w3.weight.view(torch.long)
        self._w2_packed = self.w2.weight.view(torch.long)
        self.workspace = None

    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)
        shape = (hidden_states.shape[0], self.top_k)
        if self.workspace is None or self.workspace[0].shape != shape:
            self.workspace = (
                torch.empty(shape, dtype=torch.float32, device=hidden_states.device),
                torch.empty(shape, dtype=torch.int32, device=hidden_states.device),
                torch.empty_like(hidden_states),
            )
        top_k_weights, top_k_index, output = self.workspace
        topk_sigmoid(router_logits, top_k_weights, top_k_index)
        output = self._flashinfer_moe(
            input=hidden_states,
            token_selected_experts=top_k_index,
            token_final_scales=top_k_weights,
            fc1_expert_weights=self._w1_packed,
            fc2_expert_weights=self._w2_packed,
            output_dtype=hidden_states.dtype,
            quant_scales=self._flashinfer_quant_scales,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            output=output,
            use_fused_finalize=True,
        )[0]
        return all_reduce(output)


class MiniMaxM2DecoderLayer(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.self_attn = MiniMaxM2Attention(config)
        self.block_sparse_moe = MiniMaxM2SparseMoeBlock(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)
        return hidden_states, residual


class MiniMaxM2Model(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MiniMaxM2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MiniMaxM2ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "w1": ("w1_w3", 0),
        "w3": ("w1_w3", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.model = MiniMaxM2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
