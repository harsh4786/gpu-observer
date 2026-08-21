"""Compact vLLM V1 engine-step telemetry.

The module is disabled unless GPU_OBSERVER_SEMANTIC_LIB and
GPU_OBSERVER_SEMANTIC_SHM are set. Steady-state step emission performs one
ctypes call into a preallocated shared-memory SPSC producer.
"""

from __future__ import annotations

import ctypes
import os
import re
import time
import warnings

# vLLM's own input_processor.py always appends "-{random_uuid():.8}" to the
# request_id the frontend supplies (vllm/v1/engine/input_processor.py:232,
# `request.request_id = f"{request.external_req_id}-{random_uuid():.8}"`)
# before EngineCore's scheduler ever sees it. The frontend hashes and
# publishes focus using its own un-suffixed ID (see
# gpu_observer_query_capture.py's set_live_focus), so an exact-hash match
# against EngineCore's request_id never succeeds without stripping this
# suffix first. See _resolve_focus_alias below.
_FOCUS_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$")

_PREFILL = 1
_DECODE = 2
_ENGINE_REQUEST_ADMITTED = 9

try:
    # Shared with gpu_observer_query_capture.py (frontend process) so both
    # sides hash a request's internal ID identically. Requires the new
    # vllm.gpu_observer_hash bind-mount; fall back to an inline copy so
    # existing deployments that only mount this file keep working unchanged.
    from vllm.gpu_observer_hash import stable_request_id as _stable_request_id
except ImportError:
    _FNV_OFFSET = 0xCBF29CE484222325
    _FNV_PRIME = 0x100000001B3
    _U64_MASK = (1 << 64) - 1

    def _stable_request_id(request_id: str) -> int:
        value = _FNV_OFFSET
        for byte in request_id.encode("utf-8"):
            value ^= byte
            value = (value * _FNV_PRIME) & _U64_MASK
        return value


class _Slice(ctypes.Structure):
    _fields_ = [
        ("request_id", ctypes.c_uint64),
        ("sequence_id", ctypes.c_uint64),
        ("scheduled_tokens", ctypes.c_uint32),
        ("service_class_id", ctypes.c_uint16),
        ("phase", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8),
    ]


class _PackedSlice(ctypes.Structure):
    _fields_ = [
        ("request_id", ctypes.c_uint64),
        ("packing_generation", ctypes.c_uint64),
        ("row_begin", ctypes.c_uint32),
        ("row_end", ctypes.c_uint32),
        ("scheduled_tokens", ctypes.c_uint32),
        ("packed_index", ctypes.c_uint32),
        ("phase", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 7),
    ]
class _PackedToken(ctypes.Structure):
    _fields_ = [
        ("request_id", ctypes.c_uint64),
        ("packing_generation", ctypes.c_uint64),
        ("packed_row", ctypes.c_uint32),
        ("sequence_position", ctypes.c_uint32),
        ("token_id", ctypes.c_uint32),
        ("phase", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
    ]


class _OutputToken(ctypes.Structure):
    _fields_ = [
        ("request_id", ctypes.c_uint64),
        ("output_position", ctypes.c_uint32),
        ("token_id", ctypes.c_uint32),
    ]


class _DisabledEmitter:
    __slots__ = ()

    def admitted(self, request, scheduler) -> None:
        return None

    def begin(self, scheduler_output, scheduler) -> int:
        return 0

    def packed_layout(self, step_id: int, request_ids, token_counts) -> int:
        return 0

    def packed_tokens(self, scheduler_output, request_ids, req_indices, positions, token_ids) -> None:
        return None

    def accepted_tokens(self, step_id: int, engine_core_outputs) -> None:
        return None

    def end(self, step_id: int, status: int) -> None:
        return None

    @property
    def enabled(self) -> bool:
        return False


class SemanticEmitter:
    __slots__ = (
        "_bridge",
        "_current_focus_hash",
        "_faulted",
        "_focus_output_position",
        "_focus_request_id",
        "_hash_owners",
        "_lib",
        "_max_focused_tokens",
        "_max_slices",
        "_next_packing_generation",
        "_next_step_id",
        "_packed_slices",
        "_packed_tokens",
        "_output_tokens",
        "_phases",
        "_raw_focus_hash",
        "_request_hashes",
        "_slices",
    )

    def __init__(
        self,
        library_path: str,
        shared_memory_path: str,
        capacity: int,
        max_slices: int,
        max_focused_tokens: int,
        focus_request_id: str | None,
    ) -> None:
        if max_slices < 1 or max_slices > 65_536:
            raise ValueError("max_slices must be between 1 and 65536")
        if max_focused_tokens < 1 or max_focused_tokens > 65_536:
            raise ValueError("max_focused_tokens must be between 1 and 65536")
        self._lib = ctypes.CDLL(library_path)
        self._configure_abi()
        self._bridge = self._lib.gpu_observer_bridge_open(
            os.fsencode(shared_memory_path), capacity
        )
        if not self._bridge:
            raise RuntimeError("unable to initialize semantic shared-memory bridge")

        self._faulted = False
        self._max_slices = max_slices
        # Static seed only, for backward compatibility with single-run offline
        # captures that pre-know their one request of interest (e.g. the
        # join-b production-gate experiments). Live per-request focus, set by
        # the frontend process at request-admission time via the ring's
        # focus_request_id cursor, is what actually gates emission below.
        # _raw_focus_hash mirrors the ring exactly (used only to detect a
        # genuine change -- a new chat message); _current_focus_hash is the
        # value actually compared against, rebased from _raw_focus_hash onto
        # EngineCore's real (vLLM-suffixed) request_id hash the first time
        # that request is seen -- see _resolve_focus_alias.
        self._focus_request_id = focus_request_id or None
        self._focus_output_position = 0
        self._raw_focus_hash = 0
        self._current_focus_hash = 0
        self._max_focused_tokens = max_focused_tokens
        self._slices = (_Slice * max_slices)()
        self._packed_slices = (_PackedSlice * max_slices)()
        focused_capacity = self._max_focused_tokens or 1
        self._packed_tokens = (_PackedToken * focused_capacity)()
        self._output_tokens = (_OutputToken * focused_capacity)()
        self._request_hashes: dict[str, int] = {}
        self._hash_owners: dict[int, str] = {}
        self._phases: dict[str, int] = {}
        self._next_step_id = 1
        self._next_packing_generation = 1
        if self._focus_request_id:
            seed_hash = self._intern(self._focus_request_id)
            self._lib.gpu_observer_bridge_set_focus(self._bridge, seed_hash)
            self._raw_focus_hash = seed_hash
            self._current_focus_hash = seed_hash

    def _configure_abi(self) -> None:
        self._lib.gpu_observer_bridge_open.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        self._lib.gpu_observer_bridge_open.restype = ctypes.c_void_p
        self._lib.gpu_observer_emit_step_begin.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint16,
            ctypes.POINTER(_Slice),
            ctypes.c_uint32,
        ]
        self._lib.gpu_observer_emit_step_begin.restype = ctypes.c_int32
        self._lib.gpu_observer_emit_packed_layout.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.POINTER(_PackedSlice),
            ctypes.c_uint32,
        ]
        self._lib.gpu_observer_emit_packed_layout.restype = ctypes.c_int32
        self._lib.gpu_observer_emit_packed_tokens.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(_PackedToken),
            ctypes.c_uint32,
        ]
        self._lib.gpu_observer_emit_packed_tokens.restype = ctypes.c_int32
        self._lib.gpu_observer_emit_output_tokens.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(_OutputToken),
            ctypes.c_uint32,
        ]
        self._lib.gpu_observer_emit_output_tokens.restype = ctypes.c_int32
        self._lib.gpu_observer_emit_step_end.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint8,
        ]
        self._lib.gpu_observer_emit_step_end.restype = ctypes.c_int32
        self._lib.gpu_observer_emit_lifecycle.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint8,
            ctypes.c_uint8,
        ]
        self._lib.gpu_observer_emit_lifecycle.restype = ctypes.c_int32
        self._lib.gpu_observer_bridge_close.argtypes = [ctypes.c_void_p]
        self._lib.gpu_observer_bridge_close.restype = None
        self._lib.gpu_observer_bridge_get_focus.argtypes = [ctypes.c_void_p]
        self._lib.gpu_observer_bridge_get_focus.restype = ctypes.c_uint64
        self._lib.gpu_observer_bridge_set_focus.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
        ]
        self._lib.gpu_observer_bridge_set_focus.restype = ctypes.c_int32

    @property
    def enabled(self) -> bool:
        return True

    def _intern(self, request_id: str) -> int:
        existing = self._request_hashes.get(request_id)
        if existing is not None:
            # Focus can be published after this request was first interned;
            # admission telemetry can therefore precede the next step's
            # shared-cursor refresh. Re-run the allocation-free alias check so
            # an existing suffixed EngineCore ID can still bind to the
            # frontend's unsuffixed ID.
            self._resolve_focus_alias(request_id, existing)
            return existing
        hashed = _stable_request_id(request_id)
        owner = self._hash_owners.get(hashed)
        if owner is not None and owner != request_id:
            raise RuntimeError("64-bit request ID collision")
        self._request_hashes[request_id] = hashed
        self._hash_owners[hashed] = request_id
        self._resolve_focus_alias(request_id, hashed)
        return hashed

    def _resolve_focus_alias(self, request_id: str, hashed: int) -> None:
        # The frontend hashes and publishes the un-suffixed request_id it
        # knows (see gpu_observer_query_capture.py's set_live_focus). vLLM's
        # own input_processor.py then appends "-{random_uuid():.8}" before
        # EngineCore's scheduler ever sees the request, so an exact-hash
        # match against _raw_focus_hash never succeeds on its own. The first
        # time a newly-interned request_id's stripped-suffix form hashes to
        # the raw focus value, rebase _current_focus_hash onto this request's
        # real, full hash so every later exact-hash comparison (packed
        # tokens, accepted tokens) works unchanged for the rest of this turn.
        if self._raw_focus_hash == 0 or hashed == self._current_focus_hash:
            return
        base = _FOCUS_SUFFIX_RE.sub("", request_id)
        if base != request_id and _stable_request_id(base) == self._raw_focus_hash:
            self._current_focus_hash = hashed

    def _live_focus_hash(self) -> int:
        return self._lib.gpu_observer_bridge_get_focus(self._bridge)

    def admitted(self, request, scheduler) -> None:
        if self._faulted:
            return
        try:
            request_id = request.request_id
            request_hash = self._intern(request_id)
            base_request_id = _FOCUS_SUFFIX_RE.sub("", request_id)
            base_request_hash = _stable_request_id(base_request_id)
            _running, waiting = scheduler.get_request_counts()
            result = self._lib.gpu_observer_emit_lifecycle(
                self._bridge,
                time.monotonic_ns(),
                request_hash,
                base_request_hash,
                0,
                0,
                waiting,
                _ENGINE_REQUEST_ADMITTED,
                0,
            )
            if result < 0:
                raise RuntimeError("native engine-admission emission rejected the request")
        except Exception as error:
            self._faulted = True
            warnings.warn(
                f"GPU observer semantic emitter disabled after admission failure: {error}",
                RuntimeWarning,
                stacklevel=1,
            )

    def begin(self, scheduler_output, scheduler) -> int:
        if self._faulted:
            return 0
        try:
            return self._begin(scheduler_output, scheduler)
        except Exception as error:
            self._faulted = True
            warnings.warn(
                f"GPU observer semantic emitter disabled after begin failure: {error}",
                RuntimeWarning,
                stacklevel=1,
            )
            return 0

    def _begin(self, scheduler_output, scheduler) -> int:
        scheduled = scheduler_output.num_scheduled_tokens
        slice_count = len(scheduled)
        if slice_count == 0 or slice_count > self._max_slices:
            return 0

        # Refresh once per step, not per call site: the frontend can change
        # focus to a new request between steps, and every focused-token emit
        # in this step must agree on the same value. Compared against
        # _raw_focus_hash (mirrors the ring exactly), not _current_focus_hash
        # (which _resolve_focus_alias may have rebased onto EngineCore's
        # suffixed request_id below) -- otherwise this would spuriously
        # "detect a change" and reset every single step after the rebase.
        live_focus = self._live_focus_hash()
        if live_focus != self._raw_focus_hash:
            self._raw_focus_hash = live_focus
            self._current_focus_hash = live_focus  # tentative; rebased by _intern below once the real request_id is seen
            self._focus_output_position = 0

        for request in scheduler_output.scheduled_new_reqs:
            self._intern(request.req_id)
            self._phases[request.req_id] = _PREFILL

        cached = scheduler_output.scheduled_cached_reqs
        for request_id, output_tokens in zip(
            cached.req_ids, cached.num_output_tokens, strict=True
        ):
            self._intern(request_id)
            self._phases[request_id] = _PREFILL if output_tokens == 0 else _DECODE

        prefill_tokens = 0
        decode_tokens = 0
        for index, (request_id, token_count) in enumerate(scheduled.items()):
            phase = self._phases.get(request_id, _DECODE)
            if phase == _PREFILL:
                prefill_tokens += token_count
            else:
                decode_tokens += token_count
            target = self._slices[index]
            target.request_id = self._intern(request_id)
            target.sequence_id = 0
            target.scheduled_tokens = token_count
            target.service_class_id = 0
            target.phase = phase
            target.reserved = 0

        running, waiting = scheduler.get_request_counts()
        usage = scheduler.kv_cache_manager.usage
        kv_usage = max(0, min(10_000, int(usage * 10_000)))
        step_id = self._next_step_id
        self._next_step_id += 1
        result = self._lib.gpu_observer_emit_step_begin(
            self._bridge,
            time.monotonic_ns(),
            step_id,
            scheduler_output.total_num_scheduled_tokens,
            prefill_tokens,
            decode_tokens,
            waiting,
            running + waiting,
            kv_usage,
            self._slices,
            slice_count,
        )

        for request_id in scheduler_output.finished_req_ids:
            hashed = self._request_hashes.pop(request_id, None)
            self._phases.pop(request_id, None)
            if hashed is not None:
                self._hash_owners.pop(hashed, None)

        published_step_id = step_id if result == 0 else 0
        scheduler_output._gpu_observer_step_id = published_step_id
        return published_step_id

    def packed_layout(self, step_id: int, request_ids, token_counts) -> int:
        if step_id == 0 or self._faulted:
            return 0
        try:
            return self._packed_layout(step_id, request_ids, token_counts)
        except Exception as error:
            self._faulted = True
            warnings.warn(
                f"GPU observer semantic emitter disabled after packed-layout failure: {error}",
                RuntimeWarning,
                stacklevel=1,
            )
            return 0

    def _packed_layout(self, step_id: int, request_ids, token_counts) -> int:
        slice_count = len(request_ids)
        if (
            slice_count == 0
            or slice_count > self._max_slices
            or len(token_counts) != slice_count
        ):
            raise ValueError("invalid packed-layout slice count")

        generation = self._next_packing_generation
        self._next_packing_generation += 1
        row_begin = 0
        for packed_index, (request_id, token_count_raw) in enumerate(
            zip(request_ids, token_counts, strict=True)
        ):
            token_count = int(token_count_raw)
            if token_count <= 0:
                raise ValueError("packed-layout token counts must be positive")
            row_end = row_begin + token_count
            target = self._packed_slices[packed_index]
            target.request_id = self._intern(request_id)
            target.packing_generation = generation
            target.row_begin = row_begin
            target.row_end = row_end
            target.scheduled_tokens = token_count
            target.packed_index = packed_index
            target.phase = self._phases.get(request_id, _DECODE)
            row_begin = row_end

        result = self._lib.gpu_observer_emit_packed_layout(
            self._bridge,
            time.monotonic_ns(),
            step_id,
            generation,
            row_begin,
            self._packed_slices,
            slice_count,
        )
        if result < 0:
            raise RuntimeError("native packed-layout emission rejected the batch")

        return generation if result == 0 else 0

    def packed_tokens(
        self,
        scheduler_output,
        request_ids,
        req_indices,
        positions,
        token_ids,
    ) -> None:
        if self._faulted or self._current_focus_hash == 0:
            return
        step_id = getattr(scheduler_output, "_gpu_observer_step_id", 0)
        generation = getattr(
            scheduler_output, "_gpu_observer_packing_generation", 0
        )
        if step_id == 0 or generation == 0:
            return
        try:
            live_focus = self._current_focus_hash
            focus_index = next(
                (
                    index
                    for index, request_id in enumerate(request_ids)
                    if self._intern(request_id) == live_focus
                ),
                None,
            )
            if focus_index is None:
                return
            focused_request_id = request_ids[focus_index]

            row_begin = next(
                (
                    row
                    for row, request_index in enumerate(req_indices)
                    if int(request_index) == focus_index
                ),
                None,
            )
            if row_begin is None:
                return
            row_end = row_begin
            total_rows = len(req_indices)
            while row_end < total_rows and int(req_indices[row_end]) == focus_index:
                row_end += 1
            if any(
                int(req_indices[row]) == focus_index
                for row in range(row_end, total_rows)
            ):
                raise RuntimeError("focused packed-token rows are not contiguous")

            token_count = row_end - row_begin
            if token_count > self._max_focused_tokens:
                raise RuntimeError("focused packed-token batch exceeds configured bound")
            request_hash = live_focus
            phase = self._phases.get(focused_request_id, _DECODE)
            for output_index, packed_row in enumerate(range(row_begin, row_end)):
                token_id = int(token_ids[packed_row])
                if token_id < 0 or token_id > 0xFFFFFFFF:
                    raise RuntimeError("focused token ID is outside the wire range")
                target = self._packed_tokens[output_index]
                target.request_id = request_hash
                target.packing_generation = generation
                target.packed_row = packed_row
                target.sequence_position = int(positions[packed_row])
                target.token_id = token_id
                target.phase = phase
                target.reserved[0] = 0
                target.reserved[1] = 0
                target.reserved[2] = 0

            result = self._lib.gpu_observer_emit_packed_tokens(
                self._bridge,
                time.monotonic_ns(),
                step_id,
                generation,
                self._packed_tokens,
                token_count,
            )
            if result < 0:
                raise RuntimeError("native focused packed-token emission rejected the batch")
        except Exception as error:
            self._faulted = True
            warnings.warn(
                f"GPU observer semantic emitter disabled after packed-token failure: {error}",
                RuntimeWarning,
                stacklevel=1,
            )

    def accepted_tokens(self, step_id: int, engine_core_outputs) -> None:
        if step_id == 0 or self._faulted or self._current_focus_hash == 0:
            return
        try:
            live_focus = self._current_focus_hash
            for output_group in engine_core_outputs.values():
                for output in output_group.outputs:
                    if self._intern(output.request_id) != live_focus:
                        continue
                    token_count = len(output.new_token_ids)
                    if token_count == 0:
                        return
                    if token_count > self._max_focused_tokens:
                        raise RuntimeError("focused output-token batch exceeds configured bound")
                    request_hash = live_focus
                    for index, token_id_raw in enumerate(output.new_token_ids):
                        token_id = int(token_id_raw)
                        if token_id < 0 or token_id > 0xFFFFFFFF:
                            raise RuntimeError("focused output token is outside the wire range")
                        target = self._output_tokens[index]
                        target.request_id = request_hash
                        target.output_position = self._focus_output_position + index
                        target.token_id = token_id
                    result = self._lib.gpu_observer_emit_output_tokens(
                        self._bridge,
                        time.monotonic_ns(),
                        step_id,
                        self._output_tokens,
                        token_count,
                    )
                    self._focus_output_position += token_count
                    if result < 0:
                        raise RuntimeError("native focused output-token emission rejected the batch")
                    return
        except Exception as error:
            self._faulted = True
            warnings.warn(
                f"GPU observer semantic emitter disabled after output-token failure: {error}",
                RuntimeWarning,
                stacklevel=1,
            )

    def end(self, step_id: int, status: int) -> None:
        if step_id == 0 or self._faulted:
            return
        try:
            result = self._lib.gpu_observer_emit_step_end(
                self._bridge,
                time.monotonic_ns(),
                step_id,
                status,
            )
            if result < 0:
                self._faulted = True
                warnings.warn(
                    "GPU observer semantic emitter disabled after end failure",
                    RuntimeWarning,
                    stacklevel=1,
                )
        except Exception as error:
            self._faulted = True
            warnings.warn(
                f"GPU observer semantic emitter disabled after end failure: {error}",
                RuntimeWarning,
                stacklevel=1,
            )


def _create_emitter() -> SemanticEmitter | _DisabledEmitter:
    library_path = os.environ.get("GPU_OBSERVER_SEMANTIC_LIB")
    shared_memory_path = os.environ.get("GPU_OBSERVER_SEMANTIC_SHM")
    if not library_path or not shared_memory_path:
        return _DisabledEmitter()

    try:
        max_focused_tokens = int(
            os.environ.get("GPU_OBSERVER_MAX_FOCUSED_TOKENS", "8192")
        )
        focus_request_id = os.environ.get("GPU_OBSERVER_FOCUS_REQUEST_ID")
        capacity = int(os.environ.get("GPU_OBSERVER_SEMANTIC_CAPACITY", "65536"))
        max_slices = int(os.environ.get("GPU_OBSERVER_MAX_SLICES", "1024"))
        return SemanticEmitter(
            library_path,
            shared_memory_path,
            capacity,
            max_slices,
            max_focused_tokens,
            focus_request_id,
        )
    except Exception as error:
        warnings.warn(
            f"GPU observer semantic emitter disabled: {error}",
            RuntimeWarning,
            stacklevel=1,
        )
        return _DisabledEmitter()


semantic_emitter = _create_emitter()
