import array

import torch
import torch.distributed as dist
from cuda.bindings import runtime as cudart


_POINTERS = None
_LOCAL = None
_OPENED = None


def peer_pointers():
    global _POINTERS, _LOCAL, _OPENED
    if _POINTERS is not None:
        return _POINTERS
    error, local = cudart.cudaMalloc(4096)
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(error)
    cudart.cudaMemset(local, 0, 4096)
    error, handle = cudart.cudaIpcGetMemHandle(local)
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(error)
    group = dist.new_group(backend="gloo")
    handles = [None] * 4
    dist.all_gather_object(handles, bytes(handle.reserved), group=group)
    pointers = []
    opened = []
    rank = dist.get_rank()
    for peer, raw in enumerate(handles):
        if peer == rank:
            pointers.append(local)
        else:
            handle = cudart.cudaIpcMemHandle_t()
            handle.reserved = raw
            error, pointer = cudart.cudaIpcOpenMemHandle(
                handle,
                cudart.cudaIpcMemLazyEnablePeerAccess,
            )
            if error != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(error)
            pointers.append(pointer)
            opened.append(pointer)
    _POINTERS = torch.tensor(
        array.array("Q", pointers),
        dtype=torch.int64,
        device="cuda",
    )
    _LOCAL = local
    _OPENED = opened
    return _POINTERS


def reset():
    global _POINTERS, _LOCAL, _OPENED
    if _POINTERS is None:
        return
    torch.cuda.synchronize()
    _POINTERS = None
    for pointer in _OPENED:
        cudart.cudaIpcCloseMemHandle(pointer)
    cudart.cudaFree(_LOCAL)
    _LOCAL = None
    _OPENED = None
