"""Diagnostic-only FlashAttention tensor metadata hook for EXP-0032.

This module is paired with the exact FlashAttention source shipped in the
pinned gpu-observer/vllm-sanitizer:cuda13.2 image. It reads tensor host
metadata only: no tensor contents, CUDA synchronization, or device copy.
"""

from __future__ import annotations

import functools
import hashlib
import os
import re
from pathlib import Path

_PINNED_FLASH_ATTN_SHA256 = (
    "7250e1c2bc498056685ba6fd9249f6c32fd7d9b882c5d2b9ede50fb2533a0292"
)
_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_BACKEND_INSTANCE_FLASH_ATTN = 1


def _enabled() -> bool:
    value = os.environ.get("GPU_OBSERVER_ATTENTION_TENSOR_RANGES")
    return value is not None and value not in ("", "0")


def _verify_pinned_source(module_path: str) -> None:
    path = Path(module_path)
    with path.open("rb") as source:
        observed = hashlib.file_digest(source, "sha256").hexdigest()
    if observed != _PINNED_FLASH_ATTN_SHA256:
        raise RuntimeError(
            "attention tensor overlay rejected an unpinned FlashAttention source: "
            f"{observed}"
        )


def _layer_id(layer_name: str) -> int:
    match = _LAYER_INDEX.search(layer_name)
    if match is None:
        raise RuntimeError(f"cannot resolve attention layer index from {layer_name!r}")
    # The semantic ABI reserves zero as unknown, so layer ordinals are 1-based.
    return int(match.group(1)) + 1


def install_attention_tensor_observer(semantic_emitter) -> None:
    """Wrap the pinned backend boundary once when the diagnostic arm is enabled."""

    if not _enabled():
        return

    from vllm.v1.attention.backends import flash_attn as flash_attn_module

    _verify_pinned_source(flash_attn_module.__file__)
    impl = flash_attn_module.FlashAttentionImpl
    original = impl.forward
    if getattr(original, "_gpu_observer_attention_tensor_hook", False):
        return

    @functools.wraps(original)
    def observed_forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if attn_metadata is not None:
            active_rows = int(attn_metadata.num_actual_tokens)
            if (
                active_rows <= 0
                or query.ndim < 1
                or output.ndim < 1
                or active_rows > int(query.shape[0])
                or active_rows > int(output.shape[0])
            ):
                raise RuntimeError("invalid active-row count at FlashAttention boundary")
            q_row_stride_bytes = int(query.stride(0)) * int(query.element_size())
            output_row_stride_bytes = int(output.stride(0)) * int(
                output.element_size()
            )
            semantic_emitter.attention_tensor_range(
                _BACKEND_INSTANCE_FLASH_ATTN,
                _layer_id(layer.layer_name),
                int(query.data_ptr()),
                int(output.data_ptr()),
                active_rows,
                q_row_stride_bytes,
                output_row_stride_bytes,
            )
        return original(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )

    observed_forward._gpu_observer_attention_tensor_hook = True
    impl.forward = observed_forward
