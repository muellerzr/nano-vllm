import triton
import triton.language as tl


@triton.jit
def _topk_sigmoid_kernel(
    logits,
    weights,
    indices,
    stride,
    EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
):
    token = tl.program_id(0)
    expert = tl.arange(0, EXPERTS)
    scores = tl.sigmoid(tl.load(logits + token * stride + expert))
    total = 0.0
    for slot in range(TOP_K):
        weight = tl.max(scores)
        index = tl.argmax(scores, axis=0)
        tl.store(weights + token * TOP_K + slot, weight)
        tl.store(indices + token * TOP_K + slot, index)
        total += weight
        scores = tl.where(expert == index, -float("inf"), scores)
    for slot in range(TOP_K):
        weight = tl.load(weights + token * TOP_K + slot)
        tl.store(weights + token * TOP_K + slot, weight / total)


def topk_sigmoid(logits, weights, indices):
    _topk_sigmoid_kernel[(logits.shape[0],)](
        logits,
        weights,
        indices,
        logits.stride(0),
        256,
        8,
    )
