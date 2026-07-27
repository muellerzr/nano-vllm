"""Capability and size gates for optional research mechanisms."""

from __future__ import annotations

MIN_SP_HIDDEN_SIZE_SM100 = 8192
MIN_SP_BYTES_PER_GPU_SM100 = 32 * 1024 * 1024
# On this vLLM nightly/SM120 combination, repeated native-SP requests are
# stable only for these small-shape cases.  Larger-prefill and long-context
# cases retain scheduler state in the SP path; the isolated case runner keeps
# those requests on stock vLLM instead of allowing a worker hang.
NATIVE_OVERLAP_STABLE_CASES = {
    ("prefill", 32, 16),
    ("decode", 1, 16),
}


def native_overlap_decision(
    *,
    requested: bool,
    capability_major: int,
    hidden_size: int,
    tensor_parallel_size: int,
    tokens: int,
    element_size: int = 2,
    backend_available: bool = True,
    force: bool = False,
) -> tuple[bool, str, int | None]:
    """Mirror the conservative vLLM native SP/async-TP eligibility gate."""

    if not requested:
        return False, "disabled", None
    if tensor_parallel_size <= 1:
        return False, "tp", None
    if capability_major < 9:
        return False, "capability", None
    if not backend_available:
        return False, "backend", None
    if force:
        return True, "forced", None
    if hidden_size < MIN_SP_HIDDEN_SIZE_SM100:
        return False, "hidden_size", None
    min_tokens = (
        MIN_SP_BYTES_PER_GPU_SM100 * tensor_parallel_size
    ) // (hidden_size * element_size)
    if tokens < min_tokens:
        return False, "token_threshold", min_tokens
    return True, "eligible", min_tokens


def sharded_downstream_decision(
    *,
    requested: bool,
    fused_moe_requires_full_hidden: bool,
    backend_available: bool,
    native_allreduce_rms_available: bool = False,
) -> tuple[bool, str]:
    """Use vLLM's native allreduce+RMSNorm boundary before fused MoE."""

    if not requested:
        return False, "disabled"
    if not backend_available:
        return False, "backend"
    if native_allreduce_rms_available:
        return True, "native_allreduce_rms"
    if fused_moe_requires_full_hidden:
        return False, "fused_moe_requires_full_hidden"
    return True, "eligible"


def native_overlap_case_decision(
    *,
    selected: bool,
    isolated_case: bool,
    phase: str,
    batch: int,
    context: int,
) -> tuple[bool, str]:
    """Keep native SP on the stable isolated shapes for this vLLM topology."""

    if not selected:
        return False, "base_selection_off"
    if not isolated_case:
        return False, "isolated_case_required"
    if (phase, batch, context) not in NATIVE_OVERLAP_STABLE_CASES:
        return False, "shape_repeat_unsupported"
    return True, "isolated_shape"
