"""Minimal NVTX annotate compatibility layer for the pinned NGC vLLM image.

The image's vLLM imports ``nvtx.annotate`` when
VLLM_NVTX_SCOPES_FOR_PROFILING=1, but the separate Python ``nvtx`` package is
not installed. PyTorch already exposes the CUDA NVTX push/pop API, which is all
the vLLM context-manager call sites need.
"""

from contextlib import contextmanager
from typing import Iterator

import torch


@contextmanager
def annotate(message: str, *args: object, **kwargs: object) -> Iterator[None]:
    del args, kwargs
    torch.cuda.nvtx.range_push(str(message))
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
