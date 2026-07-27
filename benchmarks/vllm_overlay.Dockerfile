ARG VLLM_BASE=vllm/vllm-openai@sha256:929e0ce173d6c2b44adabb6349ad6988710ca43ad9fd7fa82869cc55857f201d
FROM ${VLLM_BASE}

COPY --from=vllm_source . /workspace/vllm
COPY vllm_overlay/compressed_all_reduce.py \
    /workspace/vllm/vllm/distributed/compressed_all_reduce.py
COPY vllm_overlay/policy.py \
    /workspace/vllm/vllm/distributed/policy.py
COPY vllm_overlay/compressed_all_reduce.patch /tmp/compressed_all_reduce.patch

RUN patch --dry-run -d /workspace/vllm -p1 < /tmp/compressed_all_reduce.patch \
    && patch -d /workspace/vllm -p1 < /tmp/compressed_all_reduce.patch

RUN set -eu; \
    wheel=/usr/local/lib/python3.12/dist-packages/vllm; \
    cd "${wheel}"; \
    find . -type f -name '*.py' -exec cp --no-clobber --parents '{}' /workspace/vllm/vllm/ ';'; \
    for extension in \
        _C_stable_libtorch.abi3.so \
        _flashmla_C.abi3.so \
        _flashmla_extension_C.abi3.so \
        _moe_C_stable_libtorch.abi3.so \
        _qutlass_C.abi3.so \
        _rust_tool_parser.abi3.so \
        cumem_allocator.abi3.so \
        fs_io_C.abi3.so \
        spinloop.abi3.so; do \
        ln -s "${wheel}/${extension}" "/workspace/vllm/vllm/${extension}"; \
    done; \
    mkdir -p /workspace/vllm/vllm/vllm_flash_attn; \
    ln -s "${wheel}/vllm_flash_attn/_vllm_fa2_C.abi3.so" \
        /workspace/vllm/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so; \
    ln -s "${wheel}/vllm_flash_attn/_vllm_fa3_C.abi3.so" \
        /workspace/vllm/vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so; \
    cp "${wheel}/_version.py" /workspace/vllm/vllm/_version.py

ENV PYTHONPATH=/workspace/vllm
WORKDIR /workspace/vllm
