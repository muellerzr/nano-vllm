from flashinfer.norm import fused_add_rmsnorm, rmsnorm
import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, residual=None):
        if residual is None:
            return rmsnorm(x, self.weight, self.eps)
        fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
