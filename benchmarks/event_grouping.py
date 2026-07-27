from __future__ import annotations


def group_event_indices(
    tokens: list[int],
    phase: str,
    *,
    batch: int,
    context: int,
    iterations: int,
    decode_token_count: int | None = None,
) -> list[list[int]]:
    if phase not in {"prefill", "decode"}:
        raise ValueError(f"unsupported phase: {phase}")
    prompt_tokens = batch * context
    expected_decode_tokens = (
        batch if decode_token_count is None else decode_token_count
    )
    groups: list[list[int]] = []
    accumulated = 0
    current: list[int] = []
    expecting_prompt = True
    for index, token_count in enumerate(tokens):
        if expecting_prompt:
            accumulated += token_count
            current.append(index)
            if accumulated > prompt_tokens:
                raise ValueError(
                    f"prompt token count exceeded {prompt_tokens}: "
                    f"{[tokens[item] for item in current]}"
                )
            if accumulated == prompt_tokens:
                if phase == "prefill":
                    groups.append(current)
                    accumulated = 0
                    current = []
                else:
                    expecting_prompt = False
                    accumulated = 0
                    current = []
        elif token_count == expected_decode_tokens:
            groups.append([index])
            expecting_prompt = True
        else:
            raise ValueError(
                f"expected {expected_decode_tokens} decode tokens after {prompt_tokens} "
                f"prompt tokens, got {token_count}"
            )
    if expecting_prompt is False or accumulated or current:
        raise ValueError(
            f"incomplete logical request: accumulated {accumulated} "
            f"of {prompt_tokens} prompt tokens"
        )
    if len(groups) != iterations:
        raise ValueError(
            f"expected {iterations} logical requests, got {len(groups)} "
            f"from {tokens}"
        )
    return groups
