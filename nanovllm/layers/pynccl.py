import ctypes
import ctypes.util

import torch
import torch.distributed as dist


class UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


DTYPES = {
    torch.uint8: 1,
    torch.int32: 2,
    torch.int64: 4,
    torch.float16: 6,
    torch.float32: 7,
    torch.float64: 8,
    torch.bfloat16: 9,
    torch.float8_e4m3fn: 10,
}
OPS = {
    dist.ReduceOp.SUM: 0,
    dist.ReduceOp.PRODUCT: 1,
    dist.ReduceOp.MAX: 2,
    dist.ReduceOp.MIN: 3,
    dist.ReduceOp.AVG: 4,
}


class NcclCommunicator:

    def __init__(self, group, device):
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.device = torch.device(device)
        path = ctypes.util.find_library("nccl") or "libnccl.so.2"
        self.lib = ctypes.CDLL(path)
        self.lib.ncclGetErrorString.restype = ctypes.c_char_p
        self.lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(UniqueId)]
        self.lib.ncclCommInitRank.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            UniqueId,
            ctypes.c_int,
        ]
        self.lib.ncclAllReduce.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.ncclAllGather.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        unique_id = UniqueId()
        if self.rank == 0:
            self._call(self.lib.ncclGetUniqueId(ctypes.byref(unique_id)))
        tensor = torch.tensor(
            list(unique_id.internal),
            dtype=torch.int8,
        )
        src = dist.get_process_group_ranks(group)[0]
        dist.broadcast(tensor, src=src, group=group)
        for index, value in enumerate(tensor.tolist()):
            unique_id.internal[index] = value
        self.comm = ctypes.c_void_p()
        with torch.cuda.device(self.device):
            self._call(
                self.lib.ncclCommInitRank(
                    ctypes.byref(self.comm),
                    self.world_size,
                    unique_id,
                    self.rank,
                )
            )
            self.all_reduce(torch.zeros(1, device=self.device))
            torch.cuda.current_stream(self.device).synchronize()

    def _call(self, result):
        if result:
            error = self.lib.ncclGetErrorString(result).decode()
            raise RuntimeError(error)

    def _stream(self):
        return ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)

    def all_reduce(self, tensor, out=None, op=dist.ReduceOp.SUM):
        out = tensor if out is None else out
        self._call(
            self.lib.ncclAllReduce(
                tensor.data_ptr(),
                out.data_ptr(),
                tensor.numel(),
                DTYPES[tensor.dtype],
                OPS[op],
                self.comm,
                self._stream(),
            )
        )
        return out

    def all_gather(self, output, tensor):
        self._call(
            self.lib.ncclAllGather(
                tensor.data_ptr(),
                output.data_ptr(),
                tensor.numel(),
                DTYPES[tensor.dtype],
                self.comm,
                self._stream(),
            )
        )
        return output
