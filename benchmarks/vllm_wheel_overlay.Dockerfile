ARG VLLM_BASE=vllm/vllm-openai@sha256:929e0ce173d6c2b44adabb6349ad6988710ca43ad9fd7fa82869cc55857f201d
FROM ${VLLM_BASE}

COPY vllm_overlay/compressed_all_reduce.py \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/compressed_all_reduce.py
COPY vllm_overlay/policy.py \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/policy.py
COPY vllm_overlay/compressed_all_reduce.patch /tmp/compressed_all_reduce.patch

RUN patch --dry-run -d /usr/local/lib/python3.12/dist-packages -p1 \
    < /tmp/compressed_all_reduce.patch \
    && patch -d /usr/local/lib/python3.12/dist-packages -p1 \
    < /tmp/compressed_all_reduce.patch
