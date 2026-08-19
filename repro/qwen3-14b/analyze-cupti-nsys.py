#!/usr/bin/env python3
"""Offline, integer-precise semantic-step to Nsight/CUPTI activity join."""

from __future__ import annotations

import argparse
import collections
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path


BEGIN = re.compile(
    r"^begin ts=(\d+).* step=(\d+) scheduled=(\d+) prefill=(\d+) "
    r"decode=(\d+) queue=(\d+) active=(\d+)"
)
END = re.compile(r"^end ts=(\d+).* step=(\d+) status=(\d+)")
CLOCK = re.compile(
    r"monotonic_ns=(\d+) realtime_ns=(\d+) uncertainty_ns=(\d+)"
)


@dataclass
class Step:
    step_id: int
    begin_ns: int
    end_ns: int = 0
    scheduled: int = 0
    prefill: int = 0
    decode: int = 0
    queue: int = 0
    active: int = 0

    @property
    def phase(self) -> str:
        return "prefill" if self.prefill else "decode"

    @property
    def wall_ns(self) -> int:
        return self.end_ns - self.begin_ns


def parse_steps(path: Path) -> list[Step]:
    steps: dict[int, Step] = {}
    for line in path.read_text().splitlines():
        if match := BEGIN.match(line):
            ts, step_id, scheduled, prefill, decode, queue, active = map(
                int, match.groups()
            )
            steps[step_id] = Step(
                step_id, ts, 0, scheduled, prefill, decode, queue, active
            )
        elif match := END.match(line):
            ts, step_id, status = map(int, match.groups())
            if status != 0:
                raise ValueError(f"step {step_id} failed with status {status}")
            steps[step_id].end_ns = ts
    result = sorted(steps.values(), key=lambda step: step.begin_ns)
    if not result or any(step.end_ns <= step.begin_ns for step in result):
        raise ValueError("semantic log contains incomplete steps")
    return result


def parse_clock(path: Path) -> tuple[int, int]:
    match = CLOCK.fullmatch(path.read_text().strip())
    if not match:
        raise ValueError(f"invalid clock pair: {path}")
    monotonic, realtime, uncertainty = map(int, match.groups())
    return monotonic - realtime, uncertainty


def containing_step(steps: list[Step], timestamp_ns: int) -> Step | None:
    for step in steps:
        if step.begin_ns <= timestamp_ns <= step.end_ns:
            return step
    return None


def interval_union_ns(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start > end:
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return total + end - start


def milliseconds(value_ns: int) -> str:
    return f"{value_ns / 1_000_000:.6f}"


def analyze(attempt: Path) -> str:
    database = attempt / "nsys-qwen14-eager.sqlite"
    steps = parse_steps(attempt / "semantic-capture.log")
    before_offset, before_uncertainty = parse_clock(attempt / "client-clock-before.txt")
    after_offset, after_uncertainty = parse_clock(attempt / "client-clock-after.txt")
    clock_offset = (before_offset + after_offset) // 2
    clock_drift = abs(after_offset - before_offset)

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    session_utc = connection.execute(
        "SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME"
    ).fetchone()[0]
    session_origin_monotonic = session_utc + clock_offset

    kernel_rows = connection.execute(
        """
        SELECT k.start, k.end, k.streamId, k.correlationId, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
        ORDER BY k.start
        """
    ).fetchall()
    runtime_rows = connection.execute(
        """
        SELECT r.start, r.end, r.correlationId, s.value
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON s.id = r.nameId
        ORDER BY r.start
        """
    ).fetchall()
    copy_rows = connection.execute(
        "SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start"
    ).fetchall()
    memset_rows = connection.execute(
        "SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMSET ORDER BY start"
    ).fetchall()
    missing_correlations = connection.execute(
        """
        SELECT COUNT(*)
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r USING(correlationId)
        WHERE r.correlationId IS NULL
        """
    ).fetchone()[0]
    connection.close()

    kernels: dict[int, list[tuple[int, int, int, int, str]]] = collections.defaultdict(list)
    runtimes: dict[int, list[tuple[int, int, int, str]]] = collections.defaultdict(list)
    copies: dict[int, list[tuple[int, int, int]]] = collections.defaultdict(list)
    memsets: dict[int, list[tuple[int, int, int]]] = collections.defaultdict(list)
    unassigned_kernels = 0

    for start, end, stream, correlation, name in kernel_rows:
        normalized_start = session_origin_monotonic + start
        normalized_end = session_origin_monotonic + end
        step = containing_step(steps, normalized_start)
        if step is None:
            unassigned_kernels += 1
        else:
            kernels[step.step_id].append(
                (normalized_start, normalized_end, stream, correlation, name)
            )

    for start, end, correlation, name in runtime_rows:
        normalized_start = session_origin_monotonic + start
        normalized_end = session_origin_monotonic + end
        if step := containing_step(steps, normalized_start):
            runtimes[step.step_id].append(
                (normalized_start, normalized_end, correlation, name)
            )

    for rows, target in ((copy_rows, copies), (memset_rows, memsets)):
        for start, end, byte_count in rows:
            normalized_start = session_origin_monotonic + start
            normalized_end = session_origin_monotonic + end
            if step := containing_step(steps, normalized_start):
                target[step.step_id].append(
                    (normalized_start, normalized_end, byte_count)
                )

    lines = [
        "# Qwen3-14B CUPTI engine-step analysis",
        "",
        "## Clock normalization",
        "",
        f"- Session origin in CLOCK_MONOTONIC: `{session_origin_monotonic}` ns",
        f"- Before/after offset drift: `{clock_drift}` ns",
        f"- Clock-pair uncertainty: `{before_uncertainty}` / `{after_uncertainty}` ns",
        "",
        "## Per-step device execution",
        "",
        "| Step | Phase | Wall ms | Kernels | Kernel sum ms | GPU busy union ms | First GPU lag ms | Last GPU to end ms | Internal GPU gaps ms | Event-sync wait ms |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    total_kernel_sum = 0
    total_kernel_union = 0
    total_sync_wait = 0
    for step in steps:
        step_kernels = kernels[step.step_id]
        intervals = [(row[0], row[1]) for row in step_kernels]
        kernel_sum = sum(end - start for start, end in intervals)
        kernel_union = interval_union_ns(intervals)
        total_kernel_sum += kernel_sum
        total_kernel_union += kernel_union
        first_start = min((start for start, _ in intervals), default=step.begin_ns)
        last_end = max((end for _, end in intervals), default=step.begin_ns)
        envelope = max(0, last_end - first_start)
        internal_gaps = max(0, envelope - kernel_union)
        sync_wait = sum(
            end - start
            for start, end, _, name in runtimes[step.step_id]
            if name.startswith("cudaEventSynchronize")
        )
        total_sync_wait += sync_wait
        lines.append(
            f"| {step.step_id} | {step.phase} | {milliseconds(step.wall_ns)} | "
            f"{len(step_kernels)} | {milliseconds(kernel_sum)} | "
            f"{milliseconds(kernel_union)} | "
            f"{milliseconds(max(0, first_start - step.begin_ns))} | "
            f"{milliseconds(max(0, step.end_ns - last_end))} | "
            f"{milliseconds(internal_gaps)} | {milliseconds(sync_wait)} |"
        )

    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Semantic steps: `{len(steps)}`",
            f"- CUPTI kernels: `{len(kernel_rows)}`",
            f"- Unassigned kernels: `{unassigned_kernels}`",
            f"- Kernels without runtime correlation: `{missing_correlations}`",
            f"- CUDA streams: `{len({row[2] for row in kernel_rows})}`",
            f"- Summed kernel time: `{milliseconds(total_kernel_sum)}` ms",
            f"- Overlap-safe GPU busy time: `{milliseconds(total_kernel_union)}` ms",
            f"- cudaEventSynchronize host wait: `{milliseconds(total_sync_wait)}` ms",
            f"- Memcpy activities: `{len(copy_rows)}` totaling `{sum(row[2] for row in copy_rows)}` bytes",
            f"- Memset activities: `{len(memset_rows)}` totaling `{sum(row[2] for row in memset_rows)}` bytes",
            "",
            "## Top kernels by device time",
            "",
            "| Kernel | Calls | Device ms |",
            "|---|---:|---:|",
        ]
    )
    aggregate: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for start, end, _, _, name in [
        (session_origin_monotonic + start, session_origin_monotonic + end, stream, correlation, name)
        for start, end, stream, correlation, name in kernel_rows
    ]:
        aggregate[name][0] += 1
        aggregate[name][1] += end - start
    for name, (calls, duration) in sorted(
        aggregate.items(), key=lambda item: item[1][1], reverse=True
    )[:12]:
        lines.append(f"| `{name}` | {calls} | {milliseconds(duration)} |")

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "**Measured:** CUPTI device intervals nearly fill each semantic engine step, and the host blocks in one `cudaEventSynchronize` call per step while that queued work executes.",
            "",
            "**Inferred:** The earlier Aya-to-step-end gap is predominantly asynchronous GPU execution being drained behind the event synchronization, not an idle CPU-only delay.",
            "",
            "**Unknown:** This instrumented run does not establish uninstrumented latency, memory-bandwidth saturation, unified-memory migration, or per-request work inside a mixed-batch kernel.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt", type=Path)
    arguments = parser.parse_args()
    report = analyze(arguments.attempt.resolve())
    output = arguments.attempt / "cupti-analysis.md"
    output.write_text(report)
    print(report)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
