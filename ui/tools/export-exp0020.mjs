#!/usr/bin/env node

import { mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

const MAX_TEXT_BYTES = 64 * 1024 * 1024;

function fail(message) {
  throw new Error(message);
}

function boundedText(path) {
  const bytes = statSync(path).size;
  if (bytes <= 0 || bytes > MAX_TEXT_BYTES) {
    fail(`refusing ${path}: ${bytes} bytes is outside the cold-export bound`);
  }
  return readFileSync(path, "utf8");
}

function keyValues(line) {
  const values = {};
  for (const match of line.matchAll(/([a-z_]+)=([^\s]+)/g)) {
    values[match[1]] = match[2];
  }
  return values;
}

function required(values, name, context) {
  if (!(name in values)) fail(`${context} is missing ${name}`);
  return values[name];
}

function integer(values, name, context) {
  const value = Number.parseInt(required(values, name, context), 10);
  if (!Number.isSafeInteger(value)) fail(`${context} has invalid ${name}`);
  return value;
}

function bigint(values, name, context) {
  try {
    return BigInt(required(values, name, context));
  } catch {
    fail(`${context} has invalid ${name}`);
  }
}

function parseSemantic(text) {
  const steps = new Map();
  for (const [offset, raw] of text.split("\n").entries()) {
    const line = raw.trim();
    if (!line || line.startsWith("summary")) continue;
    const kind = line.split(/\s+/, 1)[0];
    const values = keyValues(line);
    if (!("step" in values)) continue;
    const stepId = integer(values, "step", `semantic line ${offset + 1}`);
    const step = steps.get(stepId) ?? {
      id: stepId,
      schedulerOrder: [],
      packedRows: [],
    };
    if (kind === "begin") {
      step.beginNs = bigint(values, "ts", line);
      step.scheduledTokens = integer(values, "scheduled", line);
      step.prefillTokens = integer(values, "prefill", line);
      step.decodeTokens = integer(values, "decode", line);
      step.queueDepth = integer(values, "queue", line);
      step.activeRequests = integer(values, "active", line);
    } else if (kind === "slice") {
      step.schedulerOrder.push({
        requestId: required(values, "request", line),
        phase: integer(values, "phase", line),
        tokens: integer(values, "tokens", line),
      });
    } else if (kind === "packed_begin") {
      step.packedNs = bigint(values, "ts", line);
      step.packingGeneration = integer(values, "generation", line);
    } else if (kind === "packed_slice") {
      const rows = required(values, "rows", line).match(/^\[(\d+),(\d+)\)$/);
      if (!rows) fail(`invalid packed row range: ${line}`);
      step.packedRows.push({
        requestId: required(values, "request", line),
        index: integer(values, "index", line),
        rowBegin: Number.parseInt(rows[1], 10),
        rowEnd: Number.parseInt(rows[2], 10),
        tokens: integer(values, "tokens", line),
        phase: integer(values, "phase", line),
      });
    } else if (kind === "end") {
      step.endNs = bigint(values, "ts", line);
    }
    steps.set(stepId, step);
  }

  const output = [...steps.values()].sort((a, b) => a.id - b.id);
  for (const step of output) {
    if (!step.beginNs || !step.packedNs || !step.endNs || step.packedRows.length === 0) {
      fail(`semantic step ${step.id} is incomplete`);
    }
    step.packedRows.sort((a, b) => a.rowBegin - b.rowBegin);
  }
  return output;
}

function parseJoinSummary(text) {
  const summaries = new Map();
  let final = null;
  for (const line of text.split("\n")) {
    if (line.startsWith("step=")) {
      const values = keyValues(line);
      summaries.set(integer(values, "step", line), values);
    } else if (line.startsWith("summary ")) {
      final = keyValues(line);
    }
  }
  if (!final || final.status !== "PASS") fail("Join-B report is missing its PASS summary");
  return { summaries, final };
}

function parseTsv(text, formatPrefix) {
  const lines = text.split("\n").filter(Boolean);
  if (!lines[0]?.startsWith(formatPrefix)) fail(`missing ${formatPrefix} header`);
  const headerIndex = lines.findIndex((line) => !line.startsWith("#"));
  if (headerIndex < 0) fail(`missing TSV columns after ${formatPrefix}`);
  const columns = lines[headerIndex].split("\t");
  return lines.slice(headerIndex + 1).filter((line) => !line.startsWith("#")).map((line) => {
    const fields = line.split("\t");
    if (fields.length !== columns.length) fail(`malformed TSV row: ${line}`);
    return Object.fromEntries(columns.map((column, index) => [column, fields[index]]));
  });
}

function parseClock(text) {
  const lines = text.split("\n");
  const pair = (name) => {
    const line = lines.find((candidate) => candidate.startsWith(`${name}\t`));
    if (!line) fail(`summary is missing ${name}`);
    const [, cuptiNs, monotonicNs, uncertaintyNs] = line.split("\t");
    return {
      cuptiNs: BigInt(cuptiNs),
      monotonicNs: BigInt(monotonicNs),
      uncertaintyNs: Number(uncertaintyNs),
    };
  };
  const start = pair("CLOCK_MONOTONIC_START");
  const end = pair("CLOCK_MONOTONIC_END");
  if (end.cuptiNs <= start.cuptiNs || end.monotonicNs <= start.monotonicNs) {
    fail("clock calibration interval is invalid");
  }
  return { start, end };
}

function normalize(timestampNs, clock) {
  if (timestampNs < clock.start.cuptiNs || timestampNs > clock.end.cuptiNs) {
    fail("CUPTI timestamp lies outside the calibration interval");
  }
  const x = timestampNs - clock.start.cuptiNs;
  const xSpan = clock.end.cuptiNs - clock.start.cuptiNs;
  const ySpan = clock.end.monotonicNs - clock.start.monotonicNs;
  return clock.start.monotonicNs + (x * ySpan) / xSpan;
}

function phaseName(step) {
  if (step.prefillTokens && step.decodeTokens) return "mixed";
  if (step.prefillTokens) return "prefill";
  return "decode";
}

function requestLabel(index) {
  return `Request ${String.fromCharCode(65 + index)}`;
}

function fixed(value, digits = 6) {
  return Number(value.toFixed(digits));
}

function main() {
  const runDir = resolve(process.argv[2] ?? fail("usage: export-exp0020.mjs RUN_DIR OUTPUT.json"));
  const outputPath = resolve(process.argv[3] ?? fail("usage: export-exp0020.mjs RUN_DIR OUTPUT.json"));
  const semantic = parseSemantic(boundedText(join(runDir, "semantic-dump.log")));
  const joinReport = parseJoinSummary(boundedText(join(runDir, "join-b-cupti.log")));
  const activityPath = join(runDir, "cupti-385.activities.tsv");
  const runtimePath = join(runDir, "cupti-385.runtime.tsv");
  const summaryPath = join(runDir, "cupti-385.summary.tsv");
  const activities = parseTsv(boundedText(activityPath), "# format=GOCUPTI01 ");
  const runtimeRows = parseTsv(boundedText(runtimePath), "# format=GOCUPTI_RUNTIME01 ");
  const summaryText = boundedText(summaryPath);
  const clock = parseClock(summaryText);

  const runtimeByCorrelation = new Map();
  for (const row of runtimeRows) {
    if (runtimeByCorrelation.has(row.correlation)) fail(`duplicate runtime correlation ${row.correlation}`);
    runtimeByCorrelation.set(row.correlation, row);
  }

  const packedTimeline = semantic.map((step) => ({ timestamp: step.packedNs, step }));
  const assignedKernels = new Map(semantic.map((step) => [step.id, []]));
  let unmatchedCorrelations = 0;
  let unassignedKernels = 0;

  for (const row of activities) {
    const runtime = runtimeByCorrelation.get(row.correlation);
    if (!runtime) {
      unmatchedCorrelations += 1;
      continue;
    }
    const runtimeStart = normalize(BigInt(runtime.start_ns), clock);
    let owner = null;
    for (const candidate of packedTimeline) {
      if (candidate.timestamp > runtimeStart) break;
      owner = runtimeStart <= candidate.step.endNs ? candidate.step : null;
    }
    if (!owner) {
      unassignedKernels += 1;
      continue;
    }
    const startNs = normalize(BigInt(row.start_ns), clock);
    const endNs = normalize(BigInt(row.end_ns), clock);
    assignedKernels.get(owner.id).push({
      startNs,
      endNs,
      runtimeStartNs: runtimeStart,
      correlation: Number(row.correlation),
      stream: Number(row.stream),
      graphId: Number(row.graph_id),
      graphNodeId: Number(row.graph_node_id),
      gridX: Number(row.grid_x),
      blockX: Number(row.block_x),
    });
  }

  if (unmatchedCorrelations !== 0 || unassignedKernels !== 0) {
    fail(`cold export lost evidence: unmatched=${unmatchedCorrelations}, unassigned=${unassignedKernels}`);
  }

  const initial = semantic.reduce((best, step) =>
    step.packedRows.length > best.packedRows.length ? step : best, semantic[0]);
  const initialIds = initial.packedRows.map((row) => row.requestId);
  const requestMap = new Map(initialIds.map((id, index) => [id, {
    id,
    label: requestLabel(index),
    initialRow: index,
    stepIds: [],
  }]));

  for (const step of semantic) {
    for (const row of step.packedRows) {
      if (!requestMap.has(row.requestId)) {
        const index = requestMap.size;
        requestMap.set(row.requestId, {
          id: row.requestId,
          label: requestLabel(index),
          initialRow: row.rowBegin,
          stepIds: [],
        });
      }
      requestMap.get(row.requestId).stepIds.push(step.id);
    }
  }

  const requestIndex = new Map([...requestMap.keys()].map((id, index) => [id, index]));
  const steps = semantic.map((step) => {
    const kernels = assignedKernels.get(step.id).sort((a, b) =>
      a.startNs < b.startNs ? -1 : a.startNs > b.startNs ? 1 : 0);
    const joinValues = joinReport.summaries.get(step.id);
    const wallNs = step.endNs - step.beginNs;
    const schedulerPositions = new Map(step.schedulerOrder.map((entry, index) => [entry.requestId, index]));
    const reorderedPositions = step.packedRows.reduce((count, row, index) =>
      count + Number(step.schedulerOrder[index]?.requestId !== row.requestId), 0);
    const gridX = kernels.reduce((largest, kernel) => Math.max(largest, kernel.gridX), step.scheduledTokens);
    const output = {
      id: step.id,
      phase: phaseName(step),
      wallMs: fixed(Number(wallNs) / 1e6, 6),
      packedAtMs: fixed(Number(step.packedNs - step.beginNs) / 1e6, 6),
      scheduledTokens: step.scheduledTokens,
      prefillTokens: step.prefillTokens,
      decodeTokens: step.decodeTokens,
      queueDepth: step.queueDepth,
      activeRequests: step.activeRequests,
      packingGeneration: step.packingGeneration,
      reorderedPositions,
      gridX,
      paddingRows: Math.max(0, gridX - step.scheduledTokens),
      schedulerOrder: step.schedulerOrder.map((entry, index) => ({
        ...entry,
        label: requestMap.get(entry.requestId)?.label ?? entry.requestId,
        position: index,
      })),
      packedRows: step.packedRows.map((row) => ({
        ...row,
        label: requestMap.get(row.requestId)?.label ?? row.requestId,
        schedulerPosition: schedulerPositions.get(row.requestId) ?? null,
        requestIndex: requestIndex.get(row.requestId) ?? -1,
      })),
      targetKernelFamily: "reshape_and_cache_flash_kernel",
      kernels: kernels.map((kernel, index) => ({
        layerOrdinal: index,
        startMs: fixed(Number(kernel.startNs - step.beginNs) / 1e6, 6),
        endMs: fixed(Number(kernel.endNs - step.beginNs) / 1e6, 6),
        durationUs: fixed(Number(kernel.endNs - kernel.startNs) / 1e3, 3),
        runtimeSubmitMs: fixed(Number(kernel.runtimeStartNs - step.beginNs) / 1e6, 6),
        correlation: kernel.correlation,
        stream: kernel.stream,
        graphId: kernel.graphId,
        graphNodeId: kernel.graphNodeId,
        gridX: kernel.gridX,
        blockX: kernel.blockX,
      })),
    };
    if (kernels.length) {
      output.firstTargetAtMs = output.kernels[0].startMs;
      output.lastTargetEndMs = Math.max(...output.kernels.map((kernel) => kernel.endMs));
      output.targetKernelSumUs = fixed(output.kernels.reduce((sum, kernel) => sum + kernel.durationUs, 0), 3);
      output.runtimeSubmitSpanMs = joinValues
        ? fixed(Number(required(joinValues, "runtime_submit_span_ns", `step ${step.id}`)) / 1e6, 6)
        : null;
    }
    return output;
  });

  const requests = [...requestMap.values()].map((request, index) => ({
    ...request,
    colorIndex: index,
    observedSteps: request.stepIds.length,
    firstStep: Math.min(...request.stepIds),
    lastStep: Math.max(...request.stepIds),
  }));
  const defaultRequest = requests.reduce((best, request) =>
    request.observedSteps > best.observedSteps ? request : best, requests[0]);
  const defaultStep = steps.find((step) =>
    step.id === 11 && step.packedRows.some((row) => row.requestId === defaultRequest.id))
    ?? steps.find((step) => step.reorderedPositions && step.packedRows.some((row) => row.requestId === defaultRequest.id))
    ?? steps.find((step) => step.packedRows.some((row) => row.requestId === defaultRequest.id));

  const exported = {
    schema: "GPU_OBSERVER_UI_01",
    experiment: {
      id: "EXP-0020",
      title: "Qwen3-14B request-to-cache-kernel correlation",
      run: "20260814T144544Z",
      model: "Qwen/Qwen3-14B",
      executionMode: "CUDA Graph",
      result: joinReport.final.status,
      semanticRecords: Number(joinReport.final.semantic_records),
      semanticLoss: Number(joinReport.final.semantic_loss),
      engineSteps: Number(joinReport.final.packed_steps),
      schedulerOrderMismatchSteps: Number(joinReport.final.scheduler_order_mismatch_steps),
      selectedKernels: Number(joinReport.final.selected_kernels),
      geometryInferredBlocks: 6680,
      paddingBlocks: Number(joinReport.final.padding_blocks),
      clockStartUncertaintyNs: Number(joinReport.final.clock_start_uncertainty_ns),
      clockEndUncertaintyNs: Number(joinReport.final.clock_end_uncertainty_ns),
      clockOffsetDriftNs: Number(joinReport.final.clock_offset_drift_ns),
      rawArtifact: "benchmarks/join-b/exp0020-cupti-ownership/20260814T144544Z",
    },
    evidence: {
      measured: [
        "EngineCore step boundaries and scheduler membership",
        "Authoritative GPUModelRunner packed-row ranges",
        "CUPTI runtime correlation IDs and actual kernel start/end",
      ],
      inferred: [
        "Per-request cache-kernel block ownership from packed rows and grid.x",
      ],
      unavailable: [
        "Client queue and network intervals",
        "Non-cache-kernel intervals in this target-filtered trace",
        "Same-run SASS block-entry callbacks",
      ],
    },
    defaults: { requestId: defaultRequest.id, stepId: defaultStep.id },
    requests,
    steps,
  };

  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, `${JSON.stringify(exported, null, 2)}\n`, "utf8");
  console.log(`wrote ${outputPath}`);
  console.log(`requests=${requests.length} steps=${steps.length} kernels=${activities.length}`);
}

main();
