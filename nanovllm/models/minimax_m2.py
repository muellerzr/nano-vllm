import torch
from torch import nn
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.compressed_collective import all_reduce
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import NVFP4GroupedLinear, QKVParallelLinear, ReplicatedLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class MiniMaxM2Attention(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        # Published MiniMax deployment uses BF16 attention; only the MoE
        # expert matrices are NVFP4.
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.q_norm = RMSNorm(self.q_size, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.kv_size, eps=config.rms_norm_eps)
        rope_params = getattr(config, "rope_parameters", {}) or {}
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=config.rotary_dim,
            max_position=config.max_position_embeddings,
            base=getattr(config, "rope_theta", rope_params.get("rope_theta", 10000)),
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q).view(-1, self.num_heads, self.head_dim)
        k = self.k_norm(k).view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        return self.o_proj(o.flatten(1, -1))


class MiniMaxM2SparseMoeBlock(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.gate = ReplicatedLinear(config.hidden_size, self.num_experts, bias=False)
        tp_size = dist.get_world_size()
        self.w1_w3 = NVFP4GroupedLinear(self.num_experts, config.hidden_size, config.intermediate_size * 2 // tp_size)
        self.w2 = NVFP4GroupedLinear(self.num_experts, config.intermediate_size // tp_size, config.hidden_size)
        self.act_fn = SiluAndMul()
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts))

    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)
        routing_weights = torch.sigmoid(router_logits.float())
        scores = routing_weights + self.e_score_correction_bias
        top_k_index = torch.topk(scores, self.top_k, dim=-1, sorted=False).indices
        top_k_weights = routing_weights.gather(1, top_k_index)
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
        active_experts = top_k_index.unique()
        assignments = [torch.where(top_k_index == expert_idx) for expert_idx in active_experts]
        max_tokens = max(token_idx.numel() for token_idx, _ in assignments)
        padded_tokens = (max_tokens + 127) // 128 * 128
        grouped_input = torch.zeros(active_experts.numel(), padded_tokens, hidden_states.shape[-1],
                                    dtype=hidden_states.dtype, device=hidden_states.device)
        group_sizes = torch.tensor([token_idx.numel() for token_idx, _ in assignments], dtype=torch.int32,
                                   device=hidden_states.device)
        for group, (token_idx, _) in enumerate(assignments):
            grouped_input[group, :token_idx.numel()] = hidden_states[token_idx]
        expert_ids = active_experts.to(torch.int32)
        grouped_output = self.w1_w3(grouped_input, expert_ids, group_sizes)
        grouped_output = self.act_fn(grouped_output)
        grouped_output = self.w2(grouped_output, expert_ids, group_sizes)
        output = torch.zeros_like(hidden_states)
        for group, (token_idx, slot_idx) in enumerate(assignments):
            y = grouped_output[group, :token_idx.numel()]
            output.index_add_(0, token_idx, y * top_k_weights[token_idx, slot_idx, None].to(y.dtype))
        if dist.get_world_size() > 1:
            all_reduce(output)
        return output


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
