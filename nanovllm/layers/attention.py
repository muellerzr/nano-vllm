import torch
import os
from torch import nn
import triton
import triton.language as tl

from nanovllm.utils.context import get_context

try:
    import flashinfer
except ImportError:
    flashinfer = None

_FLASHINFER_WORKSPACES = {}
_ATTENTION_HANDLES = {}


def _flashinfer_attention_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    attention_handle: int,
) -> None:
    context = get_context()
    if attention_handle not in _ATTENTION_HANDLES:
        raise RuntimeError(f"unknown Nano attention handle {attention_handle}")
    if context.is_prefill:
        result = context.flashinfer_prefill.run(query, key, value, out=output)
    else:
        result = context.flashinfer_decode.run(query, key, value, out=output)
    if result is not output:
        output.copy_(result[0] if isinstance(result, (tuple, list)) else result)


def _flashinfer_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    attention_handle: int,
) -> None:
    del query, key, value, output, attention_handle


try:
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="nanovllm_flashinfer_attention",
        op_func=_flashinfer_attention_op,
        mutates_args=["output"],
        fake_impl=_flashinfer_attention_fake,
    )
    _FLASHINFER_CUSTOM_OP = torch.ops.vllm.nanovllm_flashinfer_attention
except (ImportError, AttributeError, RuntimeError):
    _FLASHINFER_CUSTOM_OP = None


def _flashinfer_workspace(device):
    workspace = _FLASHINFER_WORKSPACES.get(device)
    if workspace is None:
        workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)
        _FLASHINFER_WORKSPACES[device] = workspace
    return workspace


class _FlashInferDecodeState:
    """One planned paged-decode execution shared by all decoder layers."""

    def __init__(self, wrapper):
        self.wrapper = wrapper

    def run(self, q, k_cache, v_cache, out=None):
        return self.wrapper.run(q, (k_cache, v_cache), out=out)


class _FlashInferPrefillState:
    """One planned paged-prefill execution shared by all decoder layers."""

    def __init__(self, wrapper, paged: bool):
        self.wrapper = wrapper
        self.paged = paged

    def run(self, q, k, v, out=None):
        if self.paged:
            return self.wrapper.run(q, (k, v), out=out)
        return self.wrapper.run(q, k, v, out=out)


def _paged_layout(block_tables: torch.Tensor, lengths: torch.Tensor, page_size: int):
    """Build FlashInfer's compact page layout without GPU-to-CPU round trips."""
    pages = (lengths + page_size - 1) // page_size
    max_pages = block_tables.shape[1]
    page_numbers = torch.arange(max_pages, device=block_tables.device, dtype=torch.int32)
    mask = page_numbers.unsqueeze(0) < pages.unsqueeze(1)
    indices = block_tables.masked_select(mask).to(torch.int32)
    indptr = torch.cat((torch.zeros(1, dtype=torch.int32, device=pages.device), torch.cumsum(pages, 0, dtype=torch.int32)))
    last_page_len = (lengths - 1).remainder(page_size).add(1).to(torch.int32)
    return indptr, indices, last_page_len


def make_flashinfer_decode_state(
    block_tables: torch.Tensor | None,
    context_lens: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    scale: float,
    backend: str = "auto",
):
    if flashinfer is None or backend not in ("auto", "flashinfer"):
        return None
    try:
        indptr, indices, last_page_len = _paged_layout(block_tables, context_lens, page_size)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            _flashinfer_workspace(block_tables.device), kv_layout="NHD", use_tensor_cores=True
        )
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            num_heads,
            num_kv_heads,
            head_dim,
            page_size,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
            o_data_type=q_dtype,
            sm_scale=scale,
        )
        return _FlashInferDecodeState(wrapper)
    except Exception:
        if backend == "flashinfer":
            raise
        return None


def make_flashinfer_prefill_state(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    scale: float,
    backend: str = "auto",
):
    if flashinfer is None:
        raise RuntimeError("FlashInfer is required by the minimal MiniMax engine")
    if backend not in ("auto", "flashinfer"):
        raise RuntimeError(f"Unsupported attention backend: {backend}")
    try:
        if block_tables is None:
            wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                _flashinfer_workspace(cu_seqlens_q.device), kv_layout="NHD"
            )
            wrapper.plan(
                cu_seqlens_q,
                cu_seqlens_k,
                num_heads,
                num_kv_heads,
                head_dim,
                head_dim,
                causal=True,
                sm_scale=scale,
                q_data_type=q_dtype,
                # Ragged prefill consumes freshly projected BF16 K/V; only
                # paged-cache attention reads the configured FP8 KV cache.
                kv_data_type=q_dtype,
                o_data_type=q_dtype,
            )
            return _FlashInferPrefillState(wrapper, paged=False)
        k_lengths = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        paged_indptr, paged_indices, last_page_len = _paged_layout(block_tables, k_lengths, page_size)
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            _flashinfer_workspace(block_tables.device), kv_layout="NHD"
        )
        wrapper.plan(
            cu_seqlens_q,
            paged_indptr,
            paged_indices,
            last_page_len,
            num_heads,
            num_kv_heads,
            head_dim,
            page_size,
            head_dim,
            causal=True,
            sm_scale=scale,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
            o_data_type=q_dtype,
        )
        return _FlashInferPrefillState(wrapper, paged=True)
    except Exception:
        raise


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # vLLM's published deployment stores K/V as FP8 E4M3 with scale=1.0.
    # The scatter kernel writes through the cache pointer, so Triton performs
    # the conversion in the same kernel (rather than launching a separate
    # cast for every layer and decode token).
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self._attention_handle = id(self)
        _ATTENTION_HANDLES[self._attention_handle] = self
        backend = os.getenv("NANOVLLM_ATTENTION_BACKEND", "flashinfer")
        self.attention_backend = backend
        self.flashinfer = flashinfer if backend in ("auto", "flashinfer") else None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.flashinfer_prefill is None:
                raise RuntimeError("prefill FlashInfer state was not prepared")
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            if os.getenv("NANOVLLM_TORCH_COMPILE", "0") == "1":
                if _FLASHINFER_CUSTOM_OP is None:
                    raise RuntimeError("vLLM FlashInfer custom op is unavailable")
                o = torch.empty_like(q)
                _FLASHINFER_CUSTOM_OP(q, k, v, o, self._attention_handle)
            else:
                o = context.flashinfer_prefill.run(q, k, v)
        else:
            if context.flashinfer_decode is None:
                raise RuntimeError("decode FlashInfer state was not prepared")
            if os.getenv("NANOVLLM_TORCH_COMPILE", "0") == "1":
                if _FLASHINFER_CUSTOM_OP is None:
                    raise RuntimeError("vLLM FlashInfer custom op is unavailable")
                o = torch.empty_like(q)
                _FLASHINFER_CUSTOM_OP(q, k_cache, v_cache, o, self._attention_handle)
            else:
                o = context.flashinfer_decode.run(q, k_cache, v_cache)
        return o
