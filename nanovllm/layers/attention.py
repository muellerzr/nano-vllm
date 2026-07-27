import torch
import os
from torch import nn
import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_varlen_func = flash_attn_with_kvcache = None
from nanovllm.utils.context import get_context

try:
    import flashinfer
except ImportError:
    flashinfer = None

_FLASHINFER_WORKSPACES = {}


def _flashinfer_workspace(device):
    workspace = _FLASHINFER_WORKSPACES.get(device)
    if workspace is None:
        workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)
        _FLASHINFER_WORKSPACES[device] = workspace
    return workspace


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
        backend = os.getenv("NANOVLLM_ATTENTION_BACKEND", "auto")
        self.attention_backend = backend
        self.flashinfer = flashinfer if backend in ("auto", "flashinfer") else None
        self._decode_wrapper = None
        self._prefill_wrapper = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = self._flashinfer_prefill(q, k, v, context) if self.flashinfer is not None and context.block_tables is None else None
            if o is None:
                if flash_attn_varlen_func is None:
                    raise RuntimeError("FlashInfer is required when flash-attn is unavailable")
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = self._flashinfer_decode(q, k_cache, v_cache, context) if self.flashinfer is not None else None
            if o is None:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True)
        return o

    def _flashinfer_decode(self, q, k_cache, v_cache, context):
        try:
            page_size = k_cache.shape[1]
            lengths = context.context_lens.detach().cpu().tolist()
            indices = []
            offsets = [0]
            last_page_len = []
            for row, length in zip(context.block_tables, lengths):
                pages = (length + page_size - 1) // page_size
                indices.append(row[:pages])
                offsets.append(offsets[-1] + pages)
                last_page_len.append((length - 1) % page_size + 1)
            indptr = torch.tensor(offsets, dtype=torch.int32, device=q.device)
            indices = torch.cat(indices).to(dtype=torch.int32)
            last_page_len = torch.tensor(last_page_len, dtype=torch.int32, device=q.device)
            if self._decode_wrapper is None:
                self._decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    _flashinfer_workspace(q.device), kv_layout="NHD", use_tensor_cores=True
                )
            self._decode_wrapper.plan(
                indptr, indices, last_page_len, self.num_heads, self.num_kv_heads, self.head_dim,
                page_size, q_data_type=q.dtype, kv_data_type=k_cache.dtype, o_data_type=q.dtype,
                sm_scale=self.scale,
            )
            return self._decode_wrapper.run(q, (k_cache, v_cache))
        except (RuntimeError, ValueError):
            if self.attention_backend == "flashinfer":
                raise
            return None

    def _flashinfer_prefill(self, q, k, v, context):
        try:
            if context.block_tables is None:
                # FlashInfer's single-sequence FA2 path supports the 48/2
                # grouped-query ratio used by MiniMax; its ragged wrapper
                # does not expose that ratio on every backend.
                offsets = context.cu_seqlens_q.detach().cpu().tolist()
                outputs = [flashinfer.single_prefill_with_kv_cache(
                    q[start:end], k[start:end], v[start:end], causal=True,
                    sm_scale=self.scale, backend="fa2"
                ) for start, end in zip(offsets, offsets[1:])]
                return torch.cat(outputs, dim=0)
            if self._prefill_wrapper is None:
                self._prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    _flashinfer_workspace(q.device), kv_layout="NHD"
                )
            q_offsets = context.cu_seqlens_q.detach().cpu().tolist()
            k_lengths = (context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]).detach().cpu().tolist()
            indices, kv_offsets, last_page_len = [], [0], []
            page_size = k_cache_page = self.k_cache.shape[1]
            for row, length in zip(context.block_tables, k_lengths):
                pages = (length + page_size - 1) // page_size
                indices.append(row[:pages])
                kv_offsets.append(kv_offsets[-1] + pages)
                last_page_len.append((length - 1) % page_size + 1)
            qo_indptr = context.cu_seqlens_q
            paged_indptr = torch.tensor(kv_offsets, dtype=torch.int32, device=q.device)
            paged_indices = torch.cat(indices).to(dtype=torch.int32)
            last_page_len = torch.tensor(last_page_len, dtype=torch.int32, device=q.device)
            self._prefill_wrapper.plan(
                qo_indptr, paged_indptr, paged_indices, last_page_len,
                self.num_heads, self.num_kv_heads, self.head_dim, page_size, self.head_dim,
                causal=True, sm_scale=self.scale, q_data_type=q.dtype,
                kv_data_type=self.k_cache.dtype, o_data_type=q.dtype,
            )
            return self._prefill_wrapper.run(q, (self.k_cache, self.v_cache))
        except (RuntimeError, ValueError):
            if self.attention_backend == "flashinfer":
                raise
            return None
