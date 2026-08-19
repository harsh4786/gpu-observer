#!/usr/bin/env python3
"""Aggregate matched EXP-0015 repetitions without discarding raw observations."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


ARMS = ("clean", "scheduler", "packed", "sass")
CLIENT_METRICS = (
    "duration",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "p50_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "p50_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
)
THROUGHPUT_METRICS = {
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def parse_key_values(line: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in line.strip().split():
        if "=" in token:
            key, value = token.split("=", 1)
            values[key] = value
    return values


def read_tsv(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    result: dict[str, float] = {}
    with path.open(newline="") as source:
        rows = csv.DictReader(source, delimiter="\t")
        for row in rows:
            try:
                result[row["metric"]] = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
    return result


def read_semantic_capture(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    summaries = [
        parse_key_values(line)
        for line in path.read_text().splitlines()
        if line.startswith("summary ")
    ]
    if len(summaries) != 1:
        raise ValueError(f"expected one semantic summary in {path}")
    return {key: int(value) for key, value in summaries[0].items()}


def read_step_stats(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open(newline="") as source:
        rows = csv.DictReader(
            (line for line in source if not line.startswith("#")), delimiter="\t"
        )
        for row in rows:
            if row.get("class") == "all":
                return {
                    "steps": float(row["steps"]),
                    "mean_ms": float(row["mean_ms"]),
                    "p50_ms": float(row["p50_ms"]),
                    "p95_ms": float(row["p95_ms"]),
                    "p99_ms": float(row["p99_ms"]),
                    "max_ms": float(row["max_ms"]),
                }
    raise ValueError(f"missing all-step row in {path}")


def read_join(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    summaries = [
        parse_key_values(line)
        for line in path.read_text().splitlines()
        if line.startswith("summary ")
    ]
    if len(summaries) != 1:
        raise ValueError(f"expected one Join B summary in {path}")
    return summaries[0]


def median_range(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    root = args.experiment_root.resolve(strict=True)
    runs: list[dict[str, object]] = []

    for repeat in (1, 2, 3):
        for arm in ARMS:
            run_dir = root / f"repeat-{repeat}" / arm
            with (run_dir / "client.json").open() as source:
                client = json.load(source)
            if client["completed"] != 64 or client["failed"] != 0:
                raise ValueError(f"client integrity failure: {run_dir}")
            if client["total_input_tokens"] != 8192 or client["total_output_tokens"] != 4096:
                raise ValueError(f"token-count mismatch: {run_dir}")
            if set(client["input_lens"]) != {128} or set(client["output_lens"]) != {64}:
                raise ValueError(f"non-deterministic token lengths: {run_dir}")

            semantic = read_semantic_capture(run_dir / "semantic-capture.log")
            if arm == "clean" and semantic:
                raise ValueError("clean arm unexpectedly has semantic telemetry")
            if arm != "clean":
                if semantic.get("producer_dropped") != 0:
                    raise ValueError(f"semantic loss: {run_dir}")
                if semantic.get("begins") != semantic.get("ends"):
                    raise ValueError(f"incomplete semantic steps: {run_dir}")
                if arm == "scheduler" and (
                    semantic.get("packed_begins") != 0 or semantic.get("packed_slices") != 0
                ):
                    raise ValueError(f"scheduler arm contains packed telemetry: {run_dir}")
                if arm in {"packed", "sass"} and semantic.get("packed_begins") != semantic.get("begins"):
                    raise ValueError(f"incomplete packed telemetry: {run_dir}")

            join = read_join(run_dir / "join-b-cache.log")
            if arm == "sass":
                required_zero = (
                    "semantic_loss",
                    "launches_before_packed_layout",
                    "device_drops",
                    "device_sequence_errors",
                    "orphan_events",
                    "incomplete_launches",
                    "geometry_mismatches",
                    "token_grid_mismatches",
                    "unmapped_blocks",
                    "request_count_mismatches",
                )
                if any(int(join.get(key, "-1")) != 0 for key in required_zero):
                    raise ValueError(f"Join B integrity failure: {run_dir}")
                if join.get("ownership") != "authoritative_packed_layout":
                    raise ValueError(f"non-authoritative Join B: {run_dir}")

            runs.append(
                {
                    "repeat": repeat,
                    "arm": arm,
                    "client": {key: float(client[key]) for key in CLIENT_METRICS},
                    "system": read_tsv(run_dir / "system-summary.tsv"),
                    "semantic": semantic,
                    "steps": read_step_stats(run_dir / "engine-step-stats.tsv"),
                    "join": join,
                    "artifact_path": str(run_dir.relative_to(root)),
                }
            )

    arm_stats: dict[str, dict[str, dict[str, float]]] = {}
    for arm in ARMS:
        arm_runs = [run for run in runs if run["arm"] == arm]
        arm_stats[arm] = {
            metric: median_range(
                [float(run["client"][metric]) for run in arm_runs]  # type: ignore[index]
            )
            for metric in CLIENT_METRICS
        }

    paired_deltas: dict[str, dict[str, dict[str, float]]] = {}
    for arm in ARMS[1:]:
        paired_deltas[arm] = {}
        for metric in CLIENT_METRICS:
            deltas: list[float] = []
            for repeat in (1, 2, 3):
                clean = next(
                    float(run["client"][metric])  # type: ignore[index]
                    for run in runs
                    if run["repeat"] == repeat and run["arm"] == "clean"
                )
                observed = next(
                    float(run["client"][metric])  # type: ignore[index]
                    for run in runs
                    if run["repeat"] == repeat and run["arm"] == arm
                )
                deltas.append(100.0 * (observed / clean - 1.0))
            paired_deltas[arm][metric] = {
                **median_range(deltas),
                "direction": "higher_is_better"
                if metric in THROUGHPUT_METRICS
                else "lower_is_better",
            }

    stage_pairs = (
        ("scheduler_vs_clean", "clean", "scheduler"),
        ("packed_vs_scheduler", "scheduler", "packed"),
        ("sass_vs_packed", "packed", "sass"),
    )
    incremental_deltas: dict[str, dict[str, dict[str, float]]] = {}
    for label, baseline_arm, observed_arm in stage_pairs:
        incremental_deltas[label] = {}
        for metric in CLIENT_METRICS:
            deltas = []
            for repeat in (1, 2, 3):
                baseline = next(
                    float(run["client"][metric])  # type: ignore[index]
                    for run in runs
                    if run["repeat"] == repeat and run["arm"] == baseline_arm
                )
                observed = next(
                    float(run["client"][metric])  # type: ignore[index]
                    for run in runs
                    if run["repeat"] == repeat and run["arm"] == observed_arm
                )
                deltas.append(100.0 * (observed / baseline - 1.0))
            incremental_deltas[label][metric] = {
                **median_range(deltas),
                "values": deltas,
                "direction": "higher_is_better"
                if metric in THROUGHPUT_METRICS
                else "lower_is_better",
            }

    result = {
        "experiment": "EXP-0015",
        "repetitions": 3,
        "request_shape": {
            "requests_per_run": 64,
            "input_tokens": 128,
            "output_tokens": 64,
            "concurrency": 8,
            "seed": 20260812,
        },
        "runs": runs,
        "arm_statistics": arm_stats,
        "paired_percent_deltas_vs_clean": paired_deltas,
        "incremental_percent_deltas": incremental_deltas,
    }
    args.json.write_text(json.dumps(result, indent=2) + "\n")

    def med(arm: str, metric: str) -> float:
        return arm_stats[arm][metric]["median"]

    lines = [
        "# EXP-0015 — Join B observability overhead ladder",
        "",
        "## Outcome",
        "",
        "Three counterordered fresh-server repetitions were completed per arm. Positive latency deltas are regressions; negative throughput deltas are regressions.",
        "",
        "## Median client observations",
        "",
        "| Arm | Output tok/s | TTFT p99 (ms) | ITL p99 (ms) | E2E p99 (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        lines.append(
            f"| {arm} | {med(arm, 'output_throughput'):.3f} | "
            f"{med(arm, 'p99_ttft_ms'):.3f} | {med(arm, 'p99_itl_ms'):.3f} | "
            f"{med(arm, 'p99_e2el_ms'):.3f} |"
        )
    lines += [
        "",
        "## Paired median changes versus clean",
        "",
        "| Arm | Output throughput | TTFT p99 | ITL p99 | E2E p99 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS[1:]:
        delta = paired_deltas[arm]
        lines.append(
        f"| {arm} | {delta['output_throughput']['median']:+.2f}% | "
            f"{delta['p99_ttft_ms']['median']:+.2f}% | "
            f"{delta['p99_itl_ms']['median']:+.2f}% | "
            f"{delta['p99_e2el_ms']['median']:+.2f}% |"
        )
    lines += [
        "",
        "## Incremental paired changes",
        "",
        "| Added boundary | Output throughput median [range] | TTFT p99 median [range] | ITL p99 median [range] | E2E p99 median [range] |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, _, _ in stage_pairs:
        delta = incremental_deltas[label]
        def cell(metric: str) -> str:
            values = delta[metric]
            return f"{values['median']:+.2f}% [{values['min']:+.2f}, {values['max']:+.2f}]"
        lines.append(
            f"| {label} | {cell('output_throughput')} | {cell('p99_ttft_ms')} | "
            f"{cell('p99_itl_ms')} | {cell('p99_e2el_ms')} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- Client TTFT is first streamed-token receipt minus request send, measured outside EngineCore.",
        "- Client ITL is the gap between consecutive streamed-token receipts.",
        "- Engine-step durations exist only in semantic arms and use patched Python CLOCK_MONOTONIC boundaries.",
        "- SASS events are block-entry callbacks for only `reshape_and_cache_flash_kernel`.",
        "- Three repetitions support an engineering point estimate, not a publication-grade confidence interval.",
        "",
        "Raw per-run values and delta ranges are preserved in `summary.json`.",
    ]
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
