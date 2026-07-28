import torch
from torch import nn
import triton
import triton.language as tl
import flashinfer

from nanovllm.utils.context import get_context

_FLASHINFER_WORKSPACES = {}


def _flashinfer_workspace(device):
    workspace = _FLASHINFER_WORKSPACES.get(device)
    if workspace is None:
        workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)
        _FLASHINFER_WORKSPACES[device] = workspace
    return workspace


class _FlashInferDecodeState:

    def __init__(self, wrapper):
        self.wrapper = wrapper

    def run(self, q, k_cache, v_cache):
        return self.wrapper.run(q, (k_cache, v_cache))


class _FlashInferPrefillState:

    def __init__(self, wrapper, paged: bool):
        self.wrapper = wrapper
        self.paged = paged

    def run(self, q, k, v):
        if self.paged:
            return self.wrapper.run(q, (k, v))
        return self.wrapper.run(q, k, v)


def _paged_layout(block_tables: torch.Tensor, lengths: torch.Tensor, page_size: int):
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
):
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


def make_flashinfer_prefill_state(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_tables: torch.Tensor | None,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    scale: float,
):
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
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(self):
        super().__init__()
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.flashinfer_prefill is None:
                raise RuntimeError("prefill FlashInfer state was not prepared")
            if context.block_tables is not None:
                k, v = k_cache, v_cache
            o = context.flashinfer_prefill.run(q, k, v)
        else:
            if context.flashinfer_decode is None:
                raise RuntimeError("decode FlashInfer state was not prepared")
            o = context.flashinfer_decode.run(q, k_cache, v_cache)
        return o
