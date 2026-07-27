DEFAULT_BLOCK_SIZE = 128
DEFAULT_MIN_BYTES = 3 * 1024 * 1024
FP8_MAX = 448.0


def compression_decision(
    *,
    enabled: bool,
    dtype: str,
    is_cuda: bool,
    nbytes: int,
    numel: int,
    is_contiguous: bool,
    capability_major: int,
    backend_available: bool,
    min_bytes: int = DEFAULT_MIN_BYTES,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> tuple[bool, str]:
    if not enabled:
        return False, "disabled"
    if dtype != "torch.bfloat16":
        return False, "dtype"
    if not is_cuda:
        return False, "device"
    if nbytes < min_bytes:
        return False, "threshold"
    if not is_contiguous or numel % block_size:
        return False, "layout"
    if capability_major < 9:
        return False, "capability"
    if not backend_available:
        return False, "backend"
    return True, "eligible"


def payload_bytes(
    numel: int,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    scale_bytes: int = 4,
) -> int:
    blocks = (numel + block_size - 1) // block_size
    return numel + blocks * scale_bytes
