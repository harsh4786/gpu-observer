"""Opt-in, bounded query and tokenizer manifest capture.

The serving frontend calls :func:`emit_query_manifest` only after it has the
rendered prompt and the exact token IDs that will be sent to EngineCore. The
payload is a single MessagePack Unix datagram. A missing receiver, a full socket,
or an oversized payload drops the manifest and never blocks serving.
"""

from __future__ import annotations

import ctypes
import os
import socket
import time
import warnings

import msgspec

try:
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

_SCHEMA = "GPU_OBSERVER_QUERY_01"
_DEFAULT_MAX_BYTES = 48 * 1024
_FRONTEND_REQUEST_RECEIVED = 8
_FRONTEND_TOKEN_EMITTED = 10
_FRONTEND_REQUEST_COMPLETED = 11


class _OutputToken(ctypes.Structure):
    _fields_ = [
        ("request_id", ctypes.c_uint64),
        ("output_position", ctypes.c_uint32),
        ("token_id", ctypes.c_uint32),
    ]

_warned = False



def _warn_once(message: str) -> None:
    global _warned
    if _warned:
        return
    _warned = True
    warnings.warn(message, RuntimeWarning, stacklevel=1)


class _FrontendLifecycleWriter:
    """One frontend process owns one fixed-record SPSC ring.

    Initialization may allocate and mmap. Steady-state emission only fills a
    preallocated ctypes array and crosses the native bridge; it performs no
    JSON or MessagePack serialization and never blocks serving.
    """

    __slots__ = ("_bridge", "_faulted", "_lib", "_max_tokens", "_tokens")

    def __init__(self) -> None:
        self._bridge: ctypes.c_void_p | None = None
        self._faulted = False
        self._lib: ctypes.CDLL | None = None
        self._max_tokens = 0
        self._tokens = None
        library_path = os.environ.get("GPU_OBSERVER_SEMANTIC_LIB")
        ring_path = os.environ.get("GPU_OBSERVER_FRONTEND_SHM")
        if not library_path or not ring_path:
            return
        try:
            capacity = int(os.environ.get("GPU_OBSERVER_FRONTEND_CAPACITY", "65536"))
            max_tokens = int(os.environ.get("GPU_OBSERVER_MAX_FOCUSED_TOKENS", "4096"))
            if max_tokens < 1 or max_tokens > 65_536:
                raise ValueError("GPU_OBSERVER_MAX_FOCUSED_TOKENS must be in [1, 65536]")
            lib = ctypes.CDLL(library_path)
            lib.gpu_observer_bridge_open.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
            lib.gpu_observer_bridge_open.restype = ctypes.c_void_p
            lib.gpu_observer_emit_lifecycle.argtypes = [
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
            lib.gpu_observer_emit_lifecycle.restype = ctypes.c_int32
            lib.gpu_observer_emit_lifecycle_tokens.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.POINTER(_OutputToken),
                ctypes.c_uint32,
                ctypes.c_uint8,
            ]
            lib.gpu_observer_emit_lifecycle_tokens.restype = ctypes.c_int32
            bridge = lib.gpu_observer_bridge_open(os.fsencode(ring_path), capacity)
            if not bridge:
                raise RuntimeError("unable to initialize frontend lifecycle ring")
            self._lib = lib
            self._bridge = bridge
            self._max_tokens = max_tokens
            self._tokens = (_OutputToken * max_tokens)()
        except Exception as error:
            self._faulted = True
            _warn_once(f"GPU observer frontend lifecycle disabled: {error}")

    @property
    def enabled(self) -> bool:
        return self._bridge is not None and not self._faulted

    def _single(
        self,
        timestamp_ns: int,
        request_hash: int,
        token_position: int,
        token_id: int,
        kind: int,
        status: int = 0,
    ) -> None:
        if not self.enabled:
            return
        result = self._lib.gpu_observer_emit_lifecycle(
            self._bridge,
            timestamp_ns,
            request_hash,
            0,
            token_position,
            token_id,
            0,
            kind,
            status,
        )
        if result < 0:
            raise RuntimeError("native frontend lifecycle emission rejected a record")

    def request_received(self, request_id: str, requested_tokens: int) -> int:
        request_hash = _stable_request_id(request_id)
        try:
            self._single(
                time.monotonic_ns(),
                request_hash,
                0,
                max(0, min(0xFFFFFFFF, int(requested_tokens))),
                _FRONTEND_REQUEST_RECEIVED,
            )
        except Exception as error:
            self._faulted = True
            _warn_once(f"GPU observer frontend receive event disabled: {error}")
        return request_hash

    def tokens_emitted(self, request_hash: int, first_position: int, token_ids) -> None:
        if not self.enabled or not token_ids:
            return
        try:
            token_count = len(token_ids)
            if token_count > self._max_tokens:
                raise RuntimeError("frontend token group exceeds configured bound")
            for index, token_id_raw in enumerate(token_ids):
                token_id = int(token_id_raw)
                if token_id < 0 or token_id > 0xFFFFFFFF:
                    raise RuntimeError("frontend token ID is outside the wire range")
                target = self._tokens[index]
                target.request_id = request_hash
                target.output_position = first_position + index
                target.token_id = token_id
            result = self._lib.gpu_observer_emit_lifecycle_tokens(
                self._bridge,
                time.monotonic_ns(),
                self._tokens,
                token_count,
                _FRONTEND_TOKEN_EMITTED,
            )
            if result < 0:
                raise RuntimeError("native frontend token emission rejected a group")
        except Exception as error:
            self._faulted = True
            _warn_once(f"GPU observer frontend token events disabled: {error}")

    def completed(self, request_hash: int, output_tokens: int, status: int = 0) -> None:
        try:
            self._single(
                time.monotonic_ns(),
                request_hash,
                max(0, min(0xFFFFFFFF, int(output_tokens))),
                0,
                _FRONTEND_REQUEST_COMPLETED,
                status,
            )
        except Exception as error:
            self._faulted = True
            _warn_once(f"GPU observer frontend completion event disabled: {error}")


_frontend_lifecycle = _FrontendLifecycleWriter()


def lifecycle_request_received(request_id: str, requested_tokens: int) -> int:
    return _frontend_lifecycle.request_received(request_id, requested_tokens)


def lifecycle_tokens_emitted(request_hash: int, first_position: int, token_ids) -> None:
    _frontend_lifecycle.tokens_emitted(request_hash, first_position, token_ids)


def lifecycle_request_completed(request_hash: int, output_tokens: int, status: int = 0) -> None:
    _frontend_lifecycle.completed(request_hash, output_tokens, status)


class _FocusWriter:
    """Lazily opens a non-destructive handle onto the semantic ring's shared
    header, from the vLLM frontend process, to set which request is live-
    focused for per-token detail. Independent of, and much cheaper than, the
    query/output manifest side channel below: one atomic store, no payload,
    no socket. Disabled unless GPU_OBSERVER_SEMANTIC_LIB and
    GPU_OBSERVER_SEMANTIC_SHM are set in this process's environment -- which
    they are not by default, so offline single-request captures that only
    configure GPU_OBSERVER_FOCUS_EXTERNAL_ID are unaffected."""

    def __init__(self) -> None:
        self._handle: ctypes.c_void_p | None = None
        self._lib: ctypes.CDLL | None = None
        self._attempted = False

    def _ensure_open(self) -> bool:
        if self._handle is not None:
            return True
        if self._attempted:
            return False
        self._attempted = True
        library_path = os.environ.get("GPU_OBSERVER_SEMANTIC_LIB")
        shared_memory_path = os.environ.get("GPU_OBSERVER_SEMANTIC_SHM")
        if not library_path or not shared_memory_path:
            # Intentionally silent: this is the expected, default-off state
            # (matches _DisabledEmitter's stance), not a failure.
            return False
        try:
            lib = ctypes.CDLL(library_path)
            lib.gpu_observer_bridge_focus_open.argtypes = [ctypes.c_char_p]
            lib.gpu_observer_bridge_focus_open.restype = ctypes.c_void_p
            lib.gpu_observer_bridge_focus_set.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint64,
            ]
            lib.gpu_observer_bridge_focus_set.restype = ctypes.c_int32
            handle = lib.gpu_observer_bridge_focus_open(
                os.fsencode(shared_memory_path)
            )
            if not handle:
                # Ring not created yet (EngineCore hasn't started this
                # process's bridge). Retry on the next request rather than
                # permanently disabling -- unlike SemanticEmitter, there is
                # no persistent bridge state to keep consistent here. Still
                # surfaced once: silent-forever retry with no trace at all
                # is indistinguishable from "this is broken."
                _warn_once(
                    f"GPU observer live-focus handle not yet available "
                    f"(ring not created at {shared_memory_path}?); will retry"
                )
                self._attempted = False
                return False
            self._lib = lib
            self._handle = handle
            return True
        except Exception as error:  # Telemetry must never fail a request.
            _warn_once(f"GPU observer live-focus writer disabled: {error}")
            return False

    def set_focus(self, internal_request_id: str) -> None:
        if not self._ensure_open():
            return
        try:
            self._lib.gpu_observer_bridge_focus_set(
                self._handle, _stable_request_id(internal_request_id)
            )
        except Exception as error:  # Telemetry must never fail a request.
            _warn_once(f"GPU observer live-focus set dropped: {error}")


_focus_writer = _FocusWriter()


def set_live_focus(internal_request_id: str) -> None:
    """Marks this request as the one live-focused for per-token detail.

    Single-user scope: every new request becomes the new focus unconditionally
    (no GPU_OBSERVER_FOCUS_EXTERNAL_ID gate), matching one-request-in-flight
    live demo usage rather than the single pre-known offline capture below.
    """

    _focus_writer.set_focus(internal_request_id)


def _matches_focus(request) -> bool:
    focus = os.environ.get("GPU_OBSERVER_FOCUS_EXTERNAL_ID")
    return bool(focus and getattr(request, "request_id", None) == focus)


def _token_manifest(tokenizer, rendered_prompt: str, token_ids: list[int]):
    raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    offsets = None
    offsets_verified = False

    for add_special_tokens in (False, True):
        encoded = tokenizer(
            rendered_prompt,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=True,
        )
        candidate_ids = [int(value) for value in encoded["input_ids"]]
        if candidate_ids == token_ids:
            offsets = [
                [int(begin), int(end)]
                for begin, end in encoded["offset_mapping"]
            ]
            offsets_verified = True
            break

    tokens = []
    for index, (token_id, raw_token) in enumerate(zip(token_ids, raw_tokens, strict=True)):
        offset = offsets[index] if offsets is not None else None
        display = None
        if offset is not None and offset[1] > offset[0]:
            display = rendered_prompt[offset[0] : offset[1]]
        tokens.append(
            {
                "position": index,
                "id": int(token_id),
                "raw": str(raw_token),
                "display": display,
                "offset": offset,
            }
        )
    return tokens, offsets_verified


def _send_payload(payload: bytes, socket_path: str) -> None:
    max_bytes = int(
        os.environ.get("GPU_OBSERVER_QUERY_MAX_BYTES", str(_DEFAULT_MAX_BYTES))
    )
    if max_bytes < 1024 or max_bytes > 1024 * 1024:
        raise ValueError("GPU_OBSERVER_QUERY_MAX_BYTES must be in [1024, 1048576]")
    if len(payload) > max_bytes:
        raise ValueError(
            f"query manifest is {len(payload)} bytes, above bound {max_bytes}"
        )
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        channel.setblocking(False)
        channel.sendto(payload, socket_path)
    finally:
        channel.close()


def emit_query_manifest(
    request,
    internal_request_id: str,
    rendered_prompt: str | None,
    prompt_token_ids: list[int] | None,
    tokenizer,
) -> None:
    """Emit one selected query without blocking the serving frontend."""

    socket_path = os.environ.get("GPU_OBSERVER_QUERY_SOCKET")
    if not socket_path or not _matches_focus(request):
        return
    if rendered_prompt is None or prompt_token_ids is None:
        _warn_once("GPU observer query capture skipped a prompt without text or token IDs")
        return

    try:
        token_ids = [int(value) for value in prompt_token_ids]
        tokens, offsets_verified = _token_manifest(
            tokenizer, rendered_prompt, token_ids
        )
        backend = getattr(tokenizer, "backend_tokenizer", None)
        payload = msgspec.msgpack.encode(
            {
                "schema": _SCHEMA,
                "kind": "query",
                "timestamp_ns": time.monotonic_ns(),
                "external_request_id": request.request_id,
                "internal_request_id": internal_request_id,
                "request": request.model_dump(mode="json", exclude_none=True),
                "rendered_prompt": rendered_prompt,
                "tokenizer": {
                    "module": type(tokenizer).__module__,
                    "class": type(tokenizer).__name__,
                    "is_fast": bool(getattr(tokenizer, "is_fast", False)),
                    "backend_module": (
                        type(backend).__module__ if backend is not None else None
                    ),
                    "backend_class": (
                        type(backend).__name__ if backend is not None else None
                    ),
                    "name_or_path": getattr(tokenizer, "name_or_path", None),
                },
                "prompt_token_ids": token_ids,
                "tokens": tokens,
                "offsets_verified": offsets_verified,
            }
        )
        _send_payload(payload, socket_path)
    except Exception as error:  # Telemetry must never fail a request.
        _warn_once(f"GPU observer query manifest dropped: {error}")


def emit_output_manifest(
    request, internal_request_id: str, outputs, tokenizer
) -> None:
    """Emit focused output IDs and tokenizer strings once after completion."""

    socket_path = os.environ.get("GPU_OBSERVER_QUERY_SOCKET")
    if not socket_path or not _matches_focus(request):
        return
    try:
        choices = []
        for output in outputs:
            token_ids = [int(value) for value in output.token_ids]
            raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)
            tokens = [
                {
                    "position": position,
                    "id": token_id,
                    "raw": str(raw),
                    "display": tokenizer.decode(
                        [token_id], skip_special_tokens=False
                    ),
                }
                for position, (token_id, raw) in enumerate(
                    zip(token_ids, raw_tokens, strict=True)
                )
            ]
            choices.append(
                {
                    "index": int(output.index),
                    "text": output.text,
                    "finish_reason": output.finish_reason,
                    "token_ids": token_ids,
                    "tokens": tokens,
                }
            )
        payload = msgspec.msgpack.encode(
            {
                "schema": _SCHEMA,
                "kind": "output",
                "timestamp_ns": time.monotonic_ns(),
                "external_request_id": request.request_id,
                "internal_request_id": internal_request_id,
                "choices": choices,
            }
        )
        _send_payload(payload, socket_path)
    except Exception as error:  # Telemetry must never fail a request.
        _warn_once(f"GPU observer output manifest dropped: {error}")

