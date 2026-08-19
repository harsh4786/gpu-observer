"""Experiment-only oracle for vLLM's actual packed-token row ownership.

Disabled unless GPU_OBSERVER_PACKING_ORACLE is an output path. The enabled
path emits fixed 64-byte binary records with one write per engine step. It is
an independent validation aid, not part of the production semantic stream.
"""

from __future__ import annotations

import os
import struct
import time

_MAGIC = int.from_bytes(b"GPOP", "little")
_VERSION = 1
_BEGIN = 1
_SLICE = 2
_ORDER_MATCH = 1 << 0
_COUNTS_MATCH = 1 << 1
_TOTAL_MATCH = 1 << 2
_RECORD = struct.Struct("<IHHQQQIIIIIIII")
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_U64_MASK = (1 << 64) - 1

assert _RECORD.size == 64


def _stable_request_id(request_id: str) -> int:
    value = _FNV_OFFSET
    for byte in request_id.encode("utf-8"):
        value ^= byte
        value = (value * _FNV_PRIME) & _U64_MASK
    return value


class _DisabledOracle:
    __slots__ = ()

    def observe(self, scheduler_output, req_ids, token_counts) -> None:
        return None


class PackingOracle:
    __slots__ = ("_buffer", "_fd", "_max_requests", "_view")

    def __init__(self, path: str, max_requests: int) -> None:
        if max_requests < 1 or max_requests > 65_536:
            raise ValueError("packing oracle max requests must be in [1, 65536]")
        self._max_requests = max_requests
        self._buffer = bytearray((max_requests + 1) * _RECORD.size)
        self._view = memoryview(self._buffer)
        self._fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )

    def observe(self, scheduler_output, req_ids, token_counts) -> None:
        request_count = len(req_ids)
        if request_count == 0 or request_count > self._max_requests:
            raise RuntimeError("packing oracle request count is out of bounds")
        if len(token_counts) != request_count:
            raise RuntimeError("packing oracle token-count length mismatch")

        scheduled = scheduler_output.num_scheduled_tokens
        order_matches = len(scheduled) == request_count and all(
            packed_id == scheduled_id
            for packed_id, scheduled_id in zip(req_ids, scheduled, strict=False)
        )
        counts_match = all(
            int(token_counts[index]) == int(scheduled.get(request_id, -1))
            for index, request_id in enumerate(req_ids)
        )
        total_tokens = sum(int(value) for value in token_counts)
        total_matches = total_tokens == scheduler_output.total_num_scheduled_tokens
        flags = (
            (_ORDER_MATCH if order_matches else 0)
            | (_COUNTS_MATCH if counts_match else 0)
            | (_TOTAL_MATCH if total_matches else 0)
        )
        timestamp_ns = time.monotonic_ns()
        pid = os.getpid()

        _RECORD.pack_into(
            self._buffer,
            0,
            _MAGIC,
            _VERSION,
            _BEGIN,
            timestamp_ns,
            0,
            0,
            0,
            total_tokens,
            total_tokens,
            0,
            request_count,
            total_tokens,
            flags,
            pid,
        )
        row_begin = 0
        for packed_index, (request_id, token_count_raw) in enumerate(
            zip(req_ids, token_counts, strict=True)
        ):
            token_count = int(token_count_raw)
            row_end = row_begin + token_count
            _RECORD.pack_into(
                self._buffer,
                (packed_index + 1) * _RECORD.size,
                _MAGIC,
                _VERSION,
                _SLICE,
                timestamp_ns,
                0,
                _stable_request_id(request_id),
                row_begin,
                row_end,
                token_count,
                packed_index,
                request_count,
                total_tokens,
                0,
                pid,
            )
            row_begin = row_end

        remaining = (request_count + 1) * _RECORD.size
        offset = 0
        while remaining:
            written = os.write(self._fd, self._view[offset : offset + remaining])
            if written <= 0:
                raise RuntimeError("packing oracle output write failed")
            offset += written
            remaining -= written


class _LazyOracle:
    __slots__ = ("_inner", "_max_requests", "_path")

    def __init__(self, path: str, max_requests: int) -> None:
        self._path = path
        self._max_requests = max_requests
        self._inner: PackingOracle | None = None

    def observe(self, scheduler_output, req_ids, token_counts) -> None:
        inner = self._inner
        if inner is None:
            inner = PackingOracle(self._path, self._max_requests)
            self._inner = inner
        inner.observe(scheduler_output, req_ids, token_counts)


def _create_oracle() -> _LazyOracle | _DisabledOracle:
    path = os.environ.get("GPU_OBSERVER_PACKING_ORACLE")
    if not path:
        return _DisabledOracle()
    max_requests = int(os.environ.get("GPU_OBSERVER_PACKING_MAX_REQUESTS", "1024"))
    return _LazyOracle(path, max_requests)


packing_oracle = _create_oracle()
