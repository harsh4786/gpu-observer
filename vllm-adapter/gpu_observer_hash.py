"""Shared 64-bit request-ID hash used on both sides of the semantic ring.

EngineCore (gpu_observer_semantic.py) hashes a request's internal
``chatcmpl-...`` ID before writing it into any wire record. The vLLM frontend
process (gpu_observer_query_capture.py) now also needs to hash that same
string, independently, to set the ring's live focus cursor at request
admission time -- before EngineCore ever sees the request. Both sides must
compute byte-for-byte the same value for the same string, so the function
lives once, here, instead of being duplicated and risking drift.
"""

from __future__ import annotations

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_U64_MASK = (1 << 64) - 1


def stable_request_id(request_id: str) -> int:
    value = _FNV_OFFSET
    for byte in request_id.encode("utf-8"):
        value ^= byte
        value = (value * _FNV_PRIME) & _U64_MASK
    return value
