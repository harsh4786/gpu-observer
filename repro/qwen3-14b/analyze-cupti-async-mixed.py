#!/usr/bin/env python3
"""Join production-async vLLM semantic steps to Nsight/CUPTI GPU activities."""

from __future__ import annotations

import argparse
import bisect
import collections
import re
import sqlite3
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


BEGIN = re.compile(
    r"^begin ts=(\d+).* step=(\d+) scheduled=(\d+) prefill=(\d+) "
    r"decode=(\d+) queue=(\d+) active=(\d+)"
)
SLICE = re.compile(
    r"^slice ts=(\d+).* step=(\d+) request=(0x[0-9a-f]+) phase=(\d+) tokens=(\d+)"
)
END = re.compile(r"^end ts=(\d+).* step=(\d+) status=(\d+)")
CLOCK = re.compile(
    r"monotonic_ns=(\d+) realtime_ns=(\d+) uncertainty_ns=(\d+)"
)
SCOPES = (
    "gpu_model_runner: preprocess",
    "gpu_model_runner: forward",
    "gpu_model_runner: postprocess",
    "gpu_model_runner: sample",
)
ONE_TO_ONE_SCOPES = SCOPES[1:]


@dataclass
class Step:
    step_id: int
    begin_ns: int
    scheduled: int
    prefill: int
    decode: int
    queue: int
    active: int
    end_ns: int = 0
    requests: list[tuple[str, int, int]] = field(default_factory=list)

    @property
    def phase(self) -> str:
        if self.prefill and self.decode:
            return "mixed"
        if self.prefill:
            return "prefill"
        return "decode"

    @property
    def wall_ns(self) -> int:
        return self.end_ns - self.begin_ns


@dataclass
class StepGpu:
    kernel_rows: list[tuple[int, int, int, str]]
    kernel_sum_ns: int
    busy_ns: int
    first_ns: int
    last_ns: int
    per_scope_ns: dict[str, int]


def parse_steps(path: Path) -> list[Step]:
    by_id: dict[int, Step] = {}
    order: list[Step] = []
    for line in path.read_text().splitlines():
        if match := BEGIN.match(line):
            timestamp, step_id, scheduled, prefill, decode, queue, active = map(
                int, match.groups()
            )
            step = Step(
                step_id, timestamp, scheduled, prefill, decode, queue, active
            )
            by_id[step_id] = step
            order.append(step)
        elif match := SLICE.match(line):
            _, step_id, request_id, phase, tokens = match.groups()
            by_id[int(step_id)].requests.append(
                (request_id, int(phase), int(tokens))
            )
        elif match := END.match(line):
            timestamp, step_id, status = map(int, match.groups())
            if status != 0:
                raise ValueError(f"step {step_id} failed with status {status}")
            by_id[step_id].end_ns = timestamp
    if not order or any(step.end_ns <= step.begin_ns for step in order):
        raise ValueError("semantic trace has missing or invalid step intervals")
    return order


def parse_clock(path: Path) -> tuple[int, int]:
    match = CLOCK.fullmatch(path.read_text().strip())
    if not match:
        raise ValueError(f"invalid clock pair: {path}")
    monotonic, realtime, uncertainty = map(int, match.groups())
    return monotonic - realtime, uncertainty


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


def ensure_semantic_log(attempt: Path, semantic_dump: Path) -> Path:
    output = attempt / "semantic-full.log"
    if output.is_file() and output.stat().st_size:
        return output
    with output.open("w") as stream:
        subprocess.run(
            [str(semantic_dump), str(attempt / "semantic.bin")],
            check=True,
            stdout=stream,
        )
    return output


def load_scopes(
    connection: sqlite3.Connection,
    steps: list[Step],
    session_origin_monotonic: int,
) -> dict[int, list[tuple[str, int, int, int]]]:
    scopes: dict[int, list[tuple[str, int, int, int]]] = collections.defaultdict(
        list
    )
    for name in ONE_TO_ONE_SCOPES:
        rows = connection.execute(
            "SELECT start, end, globalTid FROM NVTX_EVENTS "
            "WHERE text = ? ORDER BY start",
            (name,),
        ).fetchall()
        if len(rows) != len(steps):
            raise ValueError(
                f"{name}: expected {len(steps)} ranges, observed {len(rows)}"
            )
        for step, (start, end, tid) in zip(steps, rows):
            scopes[step.step_id].append((name, start, end, tid))

    begins = [step.begin_ns for step in steps]
    selected: set[int] = set()
    preprocess = connection.execute(
        "SELECT start, end, globalTid FROM NVTX_EVENTS "
        "WHERE text = 'gpu_model_runner: preprocess' ORDER BY start"
    ).fetchall()
    for start, end, tid in preprocess:
        normalized_start = session_origin_monotonic + start
        position = bisect.bisect_right(begins, normalized_start) - 1
        if position < 0:
            continue
        step = steps[position]
        if (
            normalized_start <= step.end_ns
            and step.step_id not in selected
        ):
            scopes[step.step_id].append(
                ("gpu_model_runner: preprocess", start, end, tid)
            )
            selected.add(step.step_id)
    if len(selected) != len(steps):
        raise ValueError(
            f"preprocess scope missing for {len(steps) - len(selected)} steps"
        )
    return scopes


def load_gpu_work(
    connection: sqlite3.Connection,
    steps: list[Step],
    scopes: dict[int, list[tuple[str, int, int, int]]],
    session_origin_monotonic: int,
) -> tuple[dict[int, StepGpu], int, int]:
    result: dict[int, StepGpu] = {}
    global_owner: dict[int, int] = {}
    for step in steps:
        unique: dict[int, tuple[int, int, str]] = {}
        per_scope: dict[str, int] = collections.defaultdict(int)
        for scope_name, start, end, tid in scopes[step.step_id]:
            rows = connection.execute(
                """
                SELECT DISTINCT k.rowid, k.start, k.end, s.value
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
                  ON r.correlationId = k.correlationId
                JOIN StringIds s ON s.id = k.shortName
                WHERE r.start >= ? AND r.start <= ? AND r.globalTid = ?
                ORDER BY k.start
                """,
                (start, end, tid),
            ).fetchall()
            for row_id, kernel_start, kernel_end, name in rows:
                owner = global_owner.setdefault(row_id, step.step_id)
                if owner != step.step_id:
                    raise ValueError(
                        f"kernel row {row_id} assigned to steps {owner} and "
                        f"{step.step_id}"
                    )
                if row_id not in unique:
                    unique[row_id] = (kernel_start, kernel_end, name)
                    per_scope[scope_name] += kernel_end - kernel_start

        kernel_rows = [
            (row_id, start, end, name)
            for row_id, (start, end, name) in unique.items()
        ]
        intervals = [
            (session_origin_monotonic + start, session_origin_monotonic + end)
            for _, start, end, _ in kernel_rows
        ]
        if not intervals:
            raise ValueError(f"step {step.step_id} has no GPU activities")
        result[step.step_id] = StepGpu(
            kernel_rows=kernel_rows,
            kernel_sum_ns=sum(end - start for start, end in intervals),
            busy_ns=interval_union_ns(intervals),
            first_ns=min(start for start, _ in intervals),
            last_ns=max(end for _, end in intervals),
            per_scope_ns=dict(per_scope),
        )
    total_kernels = connection.execute(
        "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchone()[0]
    return result, len(global_owner), total_kernels


def milliseconds(value: int) -> str:
    return f"{value / 1_000_000:.6f}"


def write_tsv(path: Path, steps: list[Step], gpu: dict[int, StepGpu]) -> None:
    header = (
        "step\tphase\tprefill_tokens\tdecode_tokens\tactive\trequests"
        "\tstep_wall_ms\tkernels\tkernel_sum_ms\tgpu_busy_union_ms"
        "\tfirst_gpu_lag_ms\tlast_gpu_to_step_end_ms\tforward_gpu_ms"
        "\tpostprocess_gpu_ms\trequest_ids"
    )
    lines = [header]
    for step in steps:
        work = gpu[step.step_id]
        request_ids = ";".join(request_id for request_id, _, _ in step.requests)
        lines.append(
            "\t".join(
                (
                    str(step.step_id),
                    step.phase,
                    str(step.prefill),
                    str(step.decode),
                    str(step.active),
                    str(len(step.requests)),
                    milliseconds(step.wall_ns),
                    str(len(work.kernel_rows)),
                    milliseconds(work.kernel_sum_ns),
                    milliseconds(work.busy_ns),
                    milliseconds(work.first_ns - step.begin_ns),
                    milliseconds(step.end_ns - work.last_ns),
                    milliseconds(
                        work.per_scope_ns.get("gpu_model_runner: forward", 0)
                    ),
                    milliseconds(
                        work.per_scope_ns.get(
                            "gpu_model_runner: postprocess", 0
                        )
                    ),
                    request_ids,
                )
            )
        )
    path.write_text("\n".join(lines) + "\n")


def write_report(
    path: Path,
    steps: list[Step],
    gpu: dict[int, StepGpu],
    assigned_kernels: int,
    total_kernels: int,
    clock_drift: int,
    uncertainty_before: int,
    uncertainty_after: int,
) -> None:
    mixed = [step for step in steps if step.phase == "mixed"]
    long_mixed = max(mixed, key=lambda step: step.prefill)
    mixed_position = steps.index(long_mixed)
    victim = steps[mixed_position + 1]
    decode = [
        step
        for step in steps
        if step.phase == "decode" and step.step_id != victim.step_id
    ]
    median_busy = int(statistics.median(gpu[step.step_id].busy_ns for step in decode))
    median_lag = int(
        statistics.median(
            gpu[step.step_id].first_ns - step.begin_ns for step in decode
        )
    )
    long_work = gpu[long_mixed.step_id]
    victim_work = gpu[victim.step_id]
    lines = [
        "# Production-async Qwen3-14B CUPTI join",
        "",
        "## Measured result",
        "",
        (
            f"- Long mixed step {long_mixed.step_id}: "
            f"{long_mixed.prefill} prefill + {long_mixed.decode} decode tokens, "
            f"{milliseconds(long_work.busy_ns)} ms GPU busy time."
        ),
        (
            f"- Median ordinary decode GPU busy time: "
            f"{milliseconds(median_busy)} ms; the long mixed step used "
            f"{long_work.busy_ns / median_busy:.2f}x as much device time."
        ),
        (
            f"- Following decode step {victim.step_id}: kernels began "
            f"{milliseconds(victim_work.first_ns - victim.begin_ns)} ms after "
            f"its scheduler begin, versus a {milliseconds(median_lag)} ms "
            f"ordinary-decode median."
        ),
        (
            f"- The following decode step's semantic interval reached "
            f"{milliseconds(victim.wall_ns)} ms."
        ),
        "",
        "## Join mechanism",
        "",
        (
            "- Semantic engine-step begins are normalized to the Nsight/CUPTI "
            "clock and paired with the one-to-one NVTX forward, postprocess, "
            "and sample ranges."
        ),
        (
            "- CUDA runtime calls inside each NVTX range supply correlation IDs; "
            "CUPTI kernel records carrying those IDs are assigned to the step."
        ),
        (
            f"- Assigned kernels: {assigned_kernels}/{total_kernels}; "
            f"unassigned: {total_kernels - assigned_kernels}."
        ),
        (
            f"- Clock offset drift: {clock_drift} ns; clock-pair uncertainty: "
            f"{uncertainty_before}/{uncertainty_after} ns."
        ),
        "",
        "## Precision boundary",
        "",
        (
            "- Step-to-kernel attribution is measured. Each mixed-step kernel "
            "still belongs to the batch as a whole; request slices are preserved "
            "as a many-to-many relationship."
        ),
        (
            "- This does not claim that a specific GPU block, warp, or instruction "
            "belongs to one request. That remains Join B."
        ),
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt", type=Path)
    parser.add_argument(
        "--semantic-dump",
        type=Path,
        default=Path("target/release/semantic_dump"),
    )
    arguments = parser.parse_args()
    attempt = arguments.attempt.resolve()
    semantic_log = ensure_semantic_log(
        attempt, arguments.semantic_dump.resolve()
    )
    steps = parse_steps(semantic_log)
    before_offset, before_uncertainty = parse_clock(
        attempt / "client-clock-before.txt"
    )
    after_offset, after_uncertainty = parse_clock(
        attempt / "client-clock-after.txt"
    )
    clock_offset = (before_offset + after_offset) // 2

    database = attempt / "nsys-qwen14-async-mixed.sqlite"
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    session_utc = connection.execute(
        "SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME"
    ).fetchone()[0]
    session_origin_monotonic = session_utc + clock_offset
    scopes = load_scopes(connection, steps, session_origin_monotonic)
    gpu, assigned, total = load_gpu_work(
        connection, steps, scopes, session_origin_monotonic
    )
    connection.close()

    write_tsv(attempt / "cupti-step-join.tsv", steps, gpu)
    write_report(
        attempt / "cupti-analysis.md",
        steps,
        gpu,
        assigned,
        total,
        abs(after_offset - before_offset),
        before_uncertainty,
        after_uncertainty,
    )
    print((attempt / "cupti-analysis.md").read_text())
    print(f"Saved: {attempt / 'cupti-step-join.tsv'}")
    print(f"Saved: {attempt / 'cupti-analysis.md'}")


if __name__ == "__main__":
    main()
