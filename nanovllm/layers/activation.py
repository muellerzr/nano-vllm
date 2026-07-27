import torch
from torch import nn
from nanovllm.utils.compile import compile_inner
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    @compile_inner
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
