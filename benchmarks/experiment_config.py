"""Pure configuration helpers for independently switchable vLLM experiments."""

from __future__ import annotations

from copy import deepcopy


def build_compilation_config(
    base: dict,
    *,
    overlap: bool = False,
    sharded_downstream: bool = False,
    sp_min_token_num: int | None = None,
) -> dict:
    """Return a vLLM compilation config with native overlap toggled."""

    config = deepcopy(base)
    pass_config = config.setdefault("pass_config", {})
    if overlap:
        pass_config["enable_sp"] = True
        pass_config["fuse_gemm_comms"] = True
        if sp_min_token_num is not None:
            pass_config["sp_min_token_num"] = sp_min_token_num
    if sharded_downstream:
        pass_config["fuse_allreduce_rms"] = True
    return config
