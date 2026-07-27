import torch
from torch import nn
import torch.distributed as dist
import os

from nanovllm.layers.attention import Attention
from nanovllm.layers.compressed_collective import all_reduce
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import NVFP4GroupedLinear, QKVParallelLinear, RowParallelLinear, VLLMRouterLinear
from nanovllm.layers.minimax_rms_norm import qk_rms_norm
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.rotary_embedding import _FLASHINFER_ROPE_OP
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

try:
    import flashinfer
except ImportError:
    flashinfer = None

_MOE_HANDLES = {}


def _flashinfer_moe_op(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    s1: torch.Tensor,
    s2: torch.Tensor,
    s3: torch.Tensor,
    s4: torch.Tensor,
    s5: torch.Tensor,
    s6: torch.Tensor,
    output: torch.Tensor,
    moe_handle: int,
) -> None:
    module = _MOE_HANDLES[moe_handle]
    result = module._flashinfer_moe(
        input=hidden_states,
        token_selected_experts=top_k_index,
        token_final_scales=top_k_weights,
        fc1_expert_weights=w1,
        fc2_expert_weights=w2,
        output_dtype=hidden_states.dtype,
        quant_scales=[s1, s2, s3, s4, s5, s6],
        tp_size=dist.get_world_size(),
        tp_rank=dist.get_rank(),
        output=output,
        use_fused_finalize=True,
    )
    if result is not output:
        output.copy_(result[0] if isinstance(result, (tuple, list)) else result)


def _flashinfer_moe_fake(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    s1: torch.Tensor,
    s2: torch.Tensor,
    s3: torch.Tensor,
    s4: torch.Tensor,
    s5: torch.Tensor,
    s6: torch.Tensor,
    output: torch.Tensor,
    moe_handle: int,
) -> None:
    del hidden_states, top_k_index, top_k_weights, w1, w2
    del s1, s2, s3, s4, s5, s6, output, moe_handle


try:
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="nanovllm_flashinfer_moe",
        op_func=_flashinfer_moe_op,
        mutates_args=["output"],
        fake_impl=_flashinfer_moe_fake,
    )
    _MOE_CUSTOM_OP = torch.ops.vllm.nanovllm_flashinfer_moe
except (ImportError, AttributeError, RuntimeError):
    _MOE_CUSTOM_OP = None

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
        self._flashinfer_rope = None
        if flashinfer is not None and os.getenv("NANOVLLM_ROPE_BACKEND", "flashinfer") == "flashinfer":
            self._flashinfer_rope = flashinfer.rope.apply_rope_with_cos_sin_cache_inplace
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
        if self._flashinfer_rope is not None:
            cache = self.rotary_emb.cos_sin_cache
            if cache.ndim == 3:
                cache = cache[:, 0]
            if os.getenv("NANOVLLM_TORCH_COMPILE", "0") == "1":
                if _FLASHINFER_ROPE_OP is None:
                    raise RuntimeError("vLLM FlashInfer RoPE custom op is unavailable")
                _FLASHINFER_ROPE_OP(positions, q, k, self.head_dim, cache, True)
            else:
                self._flashinfer_rope(positions, q, k, self.head_dim, cache, True)
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
        else:
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        return self.o_proj(o.flatten(1, -1))


class MiniMaxM2SparseMoeBlock(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.gate = VLLMRouterLinear(config.hidden_size, self.num_experts)
        tp_size = dist.get_world_size()
        self.w1_w3 = NVFP4GroupedLinear(self.num_experts, config.hidden_size, config.intermediate_size * 2 // tp_size)
        self.w2 = NVFP4GroupedLinear(self.num_experts, config.intermediate_size // tp_size, config.hidden_size)
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts))
        requested_backend = os.getenv("NANOVLLM_MOE_BACKEND", "auto")
        if requested_backend not in ("auto", "flashinfer"):
            raise RuntimeError("The minimal engine only supports vLLM FlashInfer CUTLASS MoE")
        self.moe_backend = "flashinfer_cutlass"
        self._flashinfer_moe = None
        self._fused_router = None
        try:
            from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
            self._fused_router = fused_topk
        except ImportError:
            pass
        if flashinfer is not None:
            try:
                self._flashinfer_moe = flashinfer.fused_moe.cutlass_fused_moe
            except AttributeError:
                self._flashinfer_moe = None
        if self._flashinfer_moe is None:
            raise RuntimeError("FlashInfer CUTLASS MoE is unavailable")
        self._moe_handle = id(self)
        _MOE_HANDLES[self._moe_handle] = self

    def _flashinfer_forward(self, hidden_states, top_k_index, top_k_weights):
        """Run the same SM120 NVFP4 CUTLASS expert kernel as vLLM."""
        w1, w2 = self.w1_w3, self.w2
        # These descriptors are immutable after loading.  Build them once so
        # decode does not allocate/reshape six tensors for every layer/token.
        if not hasattr(self, "_flashinfer_quant_scales"):
            self._flashinfer_quant_scales = [
                w1.input_global_scale.expand(self.num_experts),
                w1.weight_scale_inv.view(torch.int32),
                w1.weight_global_scale,
                w2.input_global_scale.expand(self.num_experts),
                w2.weight_scale_inv.view(torch.int32),
                w2.weight_global_scale,
            ]
        top_k_index = top_k_index.to(torch.int32).contiguous()
        top_k_weights = top_k_weights.contiguous()
        w1_packed = w1.weight.view(torch.long)
        w2_packed = w2.weight.view(torch.long)
        if os.getenv("NANOVLLM_TORCH_COMPILE", "0") == "1":
            if _MOE_CUSTOM_OP is None:
                raise RuntimeError("vLLM FlashInfer MoE custom op is unavailable")
            output = torch.empty_like(hidden_states)
            _MOE_CUSTOM_OP(
                hidden_states,
                top_k_index,
                top_k_weights,
                w1_packed,
                w2_packed,
                *self._flashinfer_quant_scales,
                output,
                self._moe_handle,
            )
            return output
        output = self._flashinfer_moe(
            input=hidden_states,
            token_selected_experts=top_k_index,
            token_final_scales=top_k_weights,
            fc1_expert_weights=w1_packed,
            fc2_expert_weights=w2_packed,
            output_dtype=hidden_states.dtype,
            quant_scales=self._flashinfer_quant_scales,
            tp_size=dist.get_world_size(),
            tp_rank=dist.get_rank(),
            output=torch.empty_like(hidden_states),
            use_fused_finalize=True,
        )
        # Some FlashInfer builds return the provided output in a one-element
        # list from the tracing wrapper; normalize that ABI variation here.
        return output[0] if isinstance(output, (tuple, list)) else output

    def forward(self, hidden_states):
        if self._flashinfer_moe is None or self._fused_router is None:
            raise RuntimeError("vLLM fused router and FlashInfer CUTLASS MoE are required")
        router_logits = self.gate(hidden_states)
        top_k_weights, top_k_index, _ = self._fused_router(
            hidden_states, router_logits.float(), self.top_k, True,
            indices_type=torch.int32, scoring_func="sigmoid",
        )
        output = self._flashinfer_forward(hidden_states, top_k_index, top_k_weights)
        if dist.get_world_size() > 1:
            output = all_reduce(output)
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
