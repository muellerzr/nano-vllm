"""Opt-in vLLM-style graph compilation helpers."""

import os

import torch


def compile_inner(fn):
    """Keep eager kernels compiled, but avoid nested Dynamo graphs."""
    if os.getenv("NANOVLLM_TORCH_COMPILE", "0") == "1":
        return fn
    return torch.compile(fn, dynamic=True)


def maybe_compile(model: torch.nn.Module) -> torch.nn.Module:
    if os.getenv("NANOVLLM_TORCH_COMPILE", "0") != "1":
        return model
    mode = os.getenv("NANOVLLM_TORCH_COMPILE_MODE", "default")
    if mode not in ("default", "max-autotune"):
        raise ValueError("NANOVLLM_TORCH_COMPILE_MODE must be 'default' or 'max-autotune'")
    return torch.compile(model, fullgraph=False, dynamic=False, mode=mode)
