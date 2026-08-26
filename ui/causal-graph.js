import {
  svgElement,
  truncate,
  addMarker,
  addHeading,
  addNode,
  addEdge,
  addStageChip,
  scrollToCard,
  requestColor,
  addProgressBar,
} from "./graph-primitives.js?v=graph41";
import { layerStages, QWEN3_14B, KERNEL_STAGE_COUNT } from "./kernel-graph.js?v=graph41";

function shortId(value) {
  const text = String(value ?? "");
  return text.length > 12 ? `${text.slice(0, 6)}…${text.slice(-4)}` : text;
}

function naturalQuery(query) {
  const messages = query.request?.messages;
  if (!Array.isArray(messages)) return query.rendered_prompt ?? "query unavailable";
  return messages.map((message) => {
    const content = typeof message.content === "string"
      ? message.content
      : JSON.stringify(message.content ?? "");
    return `${message.role ?? "user"}: ${content}`;
  }).join(" ");
}

function outputPreview(trace, step) {
  const accepted = step.acceptedOutputTokens ?? [];
  if (!accepted.length) return "No token accepted yet";
  const outputTokens = trace.outputManifest?.choices?.[0]?.tokens ?? [];
  const first = accepted[0];
  const token = outputTokens.find((candidate) => candidate.position === first.outputPosition);
  return token
    ? `${JSON.stringify(token.display ?? token.raw)}`
    : `${accepted.length} token${accepted.length === 1 ? "" : "s"} accepted`;
}

function deepFunction(trace, kernel) {
  if (!trace.deepReplay || !kernel) return null;
  return trace.deepReplay.functions?.find((entry) =>
    entry.name === kernel.name || (entry.match && kernel.name.includes(entry.match))) ?? null;
}

/**
 * `step` may be null: the graph still renders as soon as a query exists,
 * before any engine step has begun.
 *
 * `liveActive`: true while the current step is actually in flight (between
 * its step_begin and step_end patches).
 *
 * Two structurally different renders live here:
 * - `renderOfflineGraph`: a sealed TraceBundle with real per-kernel CUPTI
 *   data (`step.kernels.length > 0`) -- the original lane-based layout,
 *   unchanged. This has genuinely richer per-launch data (grid/block/
 *   stream/duration, request-block ownership, SASS match) that deserves its
 *   own detailed presentation, and nothing here was reported as a problem.
 * - `renderLiveGraph`: everything else (live mode, or a kernel-less offline
 *   fixture) -- request-flow context (query/tokens/scheduler/packed) is
 *   compressed into a slim strip, and the real focus is a "now executing"
 *   hero panel plus a compact history ticker (see the comment above
 *   renderLiveGraph's hero/ticker section for why this replaced an earlier
 *   node-link diagram of Qwen3's 11 per-layer kernels).
 */
export function renderCausalGraph({ svg, trace, step, kernelIndex, onKernelSelect, liveActive = false, cuptiSnapshot = null, promptTokenCount = null }) {
  if (!svg || !trace) return;
  svg.replaceChildren();

  const defs = svgElement("defs");
  addMarker(defs, "arrow-measured", "#72e3b1");
  addMarker(defs, "arrow-reconstructed", "#ffc66d");
  addMarker(defs, "arrow-matched", "#b89cff");
  addMarker(defs, "arrow-unknown", "#ff8b7a");
  const scheduler = step?.schedulerSlices ?? [];
  scheduler.forEach((slice, index) => addMarker(defs, `arrow-request-${index}`, requestColor(slice.requestId)));
  svg.append(defs);

  if (step?.kernels?.length) {
    renderOfflineGraph(svg, { trace, step, kernelIndex, onKernelSelect, liveActive, scheduler });
  } else {
    renderLiveGraph(svg, { trace, step, liveActive, cuptiSnapshot, scheduler, promptTokenCount });
  }
}

function renderOfflineGraph(svg, { trace, step, kernelIndex, onKernelSelect, liveActive, scheduler }) {
  const packed = step.packedSlices ?? [];
  const laneCount = Math.max(scheduler.length, packed.length, 1);
  const rowGap = Math.max(48, Math.min(64, 320 / laneCount));
  const firstRowY = 76;
  const graphHeight = Math.max(360, firstRowY + laneCount * rowGap + 100);
  svg.setAttribute("viewBox", `0 0 1470 ${graphHeight}`);
  svg.setAttribute("height", graphHeight);

  addHeading(svg, 18, "HUMAN QUERY", "OpenAI request");
  addHeading(svg, 210, "TOKENIZER", "exact IDs + strings");
  addHeading(svg, 405, "ENGINECORE", `step ${step.id} · ${step.phase}`);
  addHeading(svg, 650, "GPUMODELRUNNER", "packed rows");
  addHeading(svg, 900, "GPU EXECUTION", "CUPTI intervals");
  addHeading(svg, 1250, "OUTCOME", "accepted token");

  const centerY = firstRowY + (laneCount - 1) * rowGap / 2;
  const queryNode = addNode(svg, {
    x: 18, y: centerY - 26, width: 168, height: 52,
    color: "#72e3b1", title: "Foreground query",
    lines: [naturalQuery(trace.query)], badge: "measured", focus: true,
    onActivate: () => scrollToCard("query-text"),
  });
  const preview = trace.query.tokens.slice(0, 3)
    .map((token) => token.display ?? token.raw ?? `#${token.id}`)
    .join(" · ");
  const tokenNode = addNode(svg, {
    x: 210, y: centerY - 26, width: 168, height: 52,
    color: "#64c7e8", title: `${trace.query.tokens.length} prompt tokens`,
    lines: preview ? [preview] : [], badge: "measured",
    onActivate: () => scrollToCard("prompt-tokens"),
  });
  addEdge(svg, queryNode, tokenNode, { color: "#72e3b1", marker: "arrow-measured", label: "tokenize" });

  const schedulerNodes = new Map();
  scheduler.forEach((slice, index) => {
    const focus = slice.requestId === trace.focus.requestHash;
    const node = addNode(svg, {
      x: 405, y: firstRowY + index * rowGap, width: 178, height: 50,
      color: requestColor(slice.requestId),
      title: focus ? "Focused slice" : `Peer request ${index}`,
      lines: [`${slice.scheduledTokens} token${slice.scheduledTokens === 1 ? "" : "s"} · ${slice.phase}`, shortId(slice.requestId)],
      badge: focus ? "focus" : "peer", focus, pulse: focus && liveActive,
      onActivate: () => scrollToCard("scheduler-slices"),
    });
    schedulerNodes.set(slice.requestId, node);
    if (focus) {
      addEdge(svg, tokenNode, node, { color: "#72e3b1", marker: "arrow-measured", label: "admitted", flowing: liveActive });
    }
  });

  const packedNodes = new Map();
  packed.forEach((slice, index) => {
    const focus = slice.requestId === trace.focus.requestHash;
    const node = addNode(svg, {
      x: 650, y: firstRowY + index * rowGap, width: 182, height: 50,
      color: requestColor(slice.requestId),
      title: focus ? "Focused rows" : `Packed P${slice.packedIndex}`,
      lines: [`rows [${slice.rowBegin}, ${slice.rowEnd})`, `${slice.scheduledTokens} token${slice.scheduledTokens === 1 ? "" : "s"} · ${slice.phase}`],
      badge: focus ? "focus" : undefined, focus, pulse: focus && liveActive,
      onActivate: () => scrollToCard("packed-slices"),
    });
    packedNodes.set(slice.requestId, node);
    const source = schedulerNodes.get(slice.requestId);
    if (source) {
      const schedulerIndex = scheduler.findIndex((candidate) => candidate.requestId === slice.requestId);
      addEdge(svg, source, node, {
        color: requestColor(slice.requestId),
        marker: `arrow-request-${Math.max(0, schedulerIndex)}`,
        width: 1.4 + Math.min(7, slice.scheduledTokens / 4),
        kind: "measured", label: focus ? "same request" : "", flowing: focus && liveActive,
      });
    }
  });

  const kernels = step.kernels;
  const selectedIndex = Math.min(kernelIndex, kernels.length - 1);
  const selectedKernel = kernels[selectedIndex] ?? null;
  const gpuNode = addNode(svg, {
    x: 900, y: centerY - 30, width: 200, height: 68,
    color: "#64c7e8", kind: "measured",
    title: `${kernels.length} kernel${kernels.length === 1 ? "" : "s"}`,
    lines: selectedKernel ? [truncate(selectedKernel.name, 28)] : [],
    badge: "actual GPU",
    onActivate: () => onKernelSelect((selectedIndex + 1) % kernels.length),
  });
  packed.forEach((slice) => {
    const source = packedNodes.get(slice.requestId);
    if (!source) return;
    const focus = slice.requestId === trace.focus.requestHash;
    const exactOwner = selectedKernel?.requestBlockOwnership?.some((owner) => owner.requestId === slice.requestId);
    addEdge(svg, source, gpuNode, {
      color: exactOwner ? "#ffc66d" : "#72e3b1",
      marker: exactOwner ? "arrow-reconstructed" : "arrow-measured",
      width: 1.2 + Math.min(6, slice.scheduledTokens / 5),
      kind: exactOwner ? "reconstructed" : "measured",
      label: focus && exactOwner ? "validated row→block" : "", flowing: focus && liveActive,
    });
  });

  const hasOutput = step.acceptedOutputTokens.length > 0;
  const outputNode = addNode(svg, {
    x: 1250, y: centerY - 70, width: 195, height: 52,
    color: hasOutput ? "#72e3b1" : "#60756d", kind: hasOutput ? "measured" : "unknown",
    title: "Accepted output", lines: [outputPreview(trace, step)],
    badge: hasOutput ? "measured" : undefined, pulse: hasOutput && liveActive,
    onActivate: () => scrollToCard("chat-reply"),
  });
  addEdge(svg, gpuNode, outputNode, {
    color: hasOutput ? "#72e3b1" : "#60756d",
    marker: hasOutput ? "arrow-measured" : "arrow-unknown",
    kind: hasOutput ? "measured" : "unknown",
    label: hasOutput ? "accepted" : "no output yet", dashed: !hasOutput, flowing: hasOutput && liveActive,
  });

  const matchedFunction = deepFunction(trace, selectedKernel);
  const sassNode = addNode(svg, {
    x: 1250, y: centerY + 18, width: 195, height: 56,
    color: matchedFunction ? "#b89cff" : "#ff8b7a", kind: matchedFunction ? "matched" : "unknown",
    title: matchedFunction ? "SASS matched" : "SASS unavailable",
    lines: matchedFunction ? [`${matchedFunction.sass?.length ?? 0} rows`] : [],
    badge: matchedFunction ? "replay" : undefined,
    onActivate: () => scrollToCard("sass-content"),
  });
  addEdge(svg, gpuNode, sassNode, {
    color: matchedFunction ? "#b89cff" : "#ff8b7a",
    marker: matchedFunction ? "arrow-matched" : "arrow-unknown",
    kind: matchedFunction ? "matched" : "unknown", dashed: true,
  });

  const note = svgElement("text", { x: 18, y: graphHeight - 18, class: "graph-footnote" });
  note.textContent = step.schedulerOrderMismatches
    ? `${step.schedulerOrderMismatches} scheduler→packed reorder(s) — crossing edges expose it.`
    : "Request-colored edges preserve identity; width ≈ scheduled tokens.";
  svg.append(note);
}

// Replaced the node-link computation graph (a snake of 11 boxes + a
// loop-back arrow) after live user testing: reported as "not intuitive at
// all" on every axis -- couldn't tell what's executing right now, couldn't
// read the layout as a sequence, didn't parse the color/pulse convention.
// This view answers exactly one question, unmissably: what is the GPU doing
// right this instant. A big hero panel names the current stage; a small
// history ticker underneath gives just enough of "what just happened" for
// context, without requiring anyone to decode a diagram first.
const HERO_WIDTH = 1084;
const LAYER_CHIP = 12;
const LAYER_GAP = 3;
const LAYER_COUNT = 40;
const ANATOMY_BAR_HEIGHT = 10;

// Per-query kernel-call counts: one horizontal bar per DAG stage, showing
// how many times that stage has actually fired (real classified launches,
// cupti-activity.js's queryStageCounts) since this chat turn began -- not
// this step, the whole query so far, which is what makes it a genuinely
// different signal from the layer sweep above (this step's progress) or
// the DAG's per-node pulse (right now). attn collapses its 3 physical
// launches into 1 count, matching the single DAG node it fires for.
const COUNTS_ROW_H = 16;
const COUNTS_ROW_GAP = 4;
const COUNTS_LABEL_W = 150;
const COUNTS_COUNT_W = 56;
const COUNTS_BLOCK_H = KERNEL_STAGE_COUNT * COUNTS_ROW_H + (KERNEL_STAGE_COUNT - 1) * COUNTS_ROW_GAP;

// Kernel DAG: one node per real kernel-stage type within a single layer (11
// nodes -- the smallest unit where every edge is a genuine measured
// dependency, not per-launch which would be 440+ nodes, not per-layer which
// would hide the operation structure). Two rows, snake-connected (6 top
// left-to-right, 5 bottom right-to-left) so consecutive stages stay close
// together instead of wrapping across the whole width. DAG_BOTTOM_GAP is
// derived, not hand-picked, so the 5-node bottom row spans the exact same
// total width as the 6-node top row -- the row-transition edges (attn to
// o_proj) land vertically aligned instead of a long diagonal.
const DAG_NODE_W = 160;
const DAG_NODE_H = 56;
const DAG_NODE_GAP = 18;
const DAG_ROW_GAP = 44;
const DAG_TOP_COUNT = 6;
const DAG_BOTTOM_COUNT = 5;
const DAG_TOP_WIDTH = DAG_TOP_COUNT * DAG_NODE_W + (DAG_TOP_COUNT - 1) * DAG_NODE_GAP;
const DAG_BOTTOM_GAP = (DAG_TOP_WIDTH - DAG_BOTTOM_COUNT * DAG_NODE_W) / (DAG_BOTTOM_COUNT - 1);
const DAG_HEIGHT = DAG_NODE_H * 2 + DAG_ROW_GAP;

function dagNodePosition(index, originX, originY) {
  if (index < DAG_TOP_COUNT) {
    return { x: originX + index * (DAG_NODE_W + DAG_NODE_GAP), y: originY };
  }
  const bottomIndex = index - DAG_TOP_COUNT;
  const slotFromLeft = (DAG_BOTTOM_COUNT - 1) - bottomIndex; // o_proj (first bottom stage) sits rightmost, under attn
  return { x: originX + slotFromLeft * (DAG_NODE_W + DAG_BOTTOM_GAP), y: originY + DAG_NODE_H + DAG_ROW_GAP };
}

// Most stage-to-stage edges are a normal left-to-right hop within one row;
// the one row transition (attn -> o_proj) is actually a vertical hop
// (attn is top-row-rightmost, o_proj is bottom-row-rightmost, directly
// below it) -- addEdge's left-to-right bezier convention would draw that as
// a long unnecessary diagonal, so detect the vertical case and draw a
// straight vertical bezier instead.
function connectDagStage(svg, from, to, options) {
  const verticalTransition = Math.abs(from.cx - to.cx) < DAG_NODE_W && to.top >= from.bottom - 4;
  if (!verticalTransition) {
    addEdge(svg, from, to, options);
    return;
  }
  const bendY = (from.bottom + to.top) / 2;
  const path = svgElement("path", {
    d: `M ${from.cx} ${from.bottom} C ${from.cx} ${bendY}, ${to.cx} ${bendY}, ${to.cx} ${to.top}`,
    class: `graph-edge ${options.kind ?? "measured"}${options.flowing ? " flowing" : ""}`,
    stroke: options.color ?? "#72e3b1",
    "stroke-width": Math.max(1.2, options.width ?? 1.4),
    "marker-end": `url(#${options.marker ?? "arrow-measured"})`,
  });
  svg.insertBefore(path, svg.querySelector(".graph-node"));
}

function renderLiveGraph(svg, { trace, step, liveActive, cuptiSnapshot, scheduler, promptTokenCount }) {
  const packed = step?.packedSlices ?? [];
  const graphWidth = 18 + HERO_WIDTH + 24;
  const heroX = 18;
  const heroY = 76;
  const layerY = heroY + DAG_HEIGHT + 40;
  const countsY = layerY + LAYER_CHIP + 34;
  const anatomyY = countsY + COUNTS_BLOCK_H + 34;
  const graphHeight = anatomyY + ANATOMY_BAR_HEIGHT + 44;
  svg.setAttribute("viewBox", `0 0 ${graphWidth} ${graphHeight}`);
  svg.setAttribute("height", graphHeight);

  // ---- Compact upstream request-flow strip -----------------------------
  // De-emphasized on purpose: real context for what's being processed, but
  // secondary to the kernel graph below, which is the actual point of this
  // view -- active execution, focused on the GPU side.
  const stripY = 8;
  const stripH = 40;
  const focusSlice = scheduler.find((slice) => slice.requestId === trace.focus.requestHash);
  const focusPacked = packed.find((slice) => slice.requestId === trace.focus.requestHash);
  const peerCount = Math.max(scheduler.length, packed.length, 1) - 1;

  const queryNode = addNode(svg, {
    x: 18, y: stripY, width: 210, height: stripH,
    color: "#72e3b1", kind: "measured", title: "Query",
    lines: [truncate(naturalQuery(trace.query), 40)],
    onActivate: () => scrollToCard("query-text"),
  });
  // promptTokenCount: the real tokenized prompt length, captured off this
  // turn's first request_slice patch (see resolveLiveFocus in trace.js) --
  // not trace.query.tokens.length, which is always empty in live mode
  // (nothing tokenizes client-side; there's no offline TraceBundle here).
  // Falls back to trace.query.tokens.length for offline fixtures, and shows
  // "unknown" styling before that first patch lands for this turn.
  const tokenNode = addNode(svg, {
    x: 242, y: stripY, width: 130, height: stripH,
    color: promptTokenCount != null ? "#64c7e8" : "#60756d",
    kind: promptTokenCount != null ? "measured" : "unknown",
    title: promptTokenCount != null ? `${promptTokenCount} tokens` : `${trace.query.tokens.length} tokens`,
    onActivate: () => scrollToCard("prompt-tokens"),
  });
  addEdge(svg, queryNode, tokenNode, { color: "#72e3b1", marker: "arrow-measured", width: 1.2 });

  let lastStripNode = tokenNode;
  if (step) {
    const schedNode = addNode(svg, {
      x: 386, y: stripY, width: 190, height: stripH,
      color: focusSlice ? requestColor(focusSlice.requestId) : "#60756d",
      kind: focusSlice ? "measured" : "unknown",
      title: focusSlice ? `step ${step.id} · ${focusSlice.phase}` : `step ${step.id}`,
      lines: peerCount > 0 ? [`+${peerCount} peer request${peerCount === 1 ? "" : "s"}`] : [],
      pulse: Boolean(focusSlice) && liveActive,
      onActivate: () => scrollToCard("scheduler-slices"),
    });
    addEdge(svg, tokenNode, schedNode, { color: "#72e3b1", marker: "arrow-measured", width: 1.2, flowing: liveActive });
    lastStripNode = schedNode;

    if (focusPacked) {
      const packedNode = addNode(svg, {
        x: 592, y: stripY, width: 190, height: stripH,
        color: requestColor(focusPacked.requestId), kind: "measured",
        title: `rows [${focusPacked.rowBegin}, ${focusPacked.rowEnd})`,
        pulse: liveActive,
        onActivate: () => scrollToCard("packed-slices"),
      });
      addEdge(svg, schedNode, packedNode, { color: "#72e3b1", marker: "arrow-measured", width: 1.2, flowing: liveActive });
      lastStripNode = packedNode;
    }
  }

  if (!step) {
    const note = svgElement("text", { x: 18, y: graphHeight - 18, class: "graph-footnote" });
    note.textContent = "Waiting for the scheduler to admit this request…";
    svg.append(note);
    return;
  }

  // ---- The main event: what is the GPU doing, as a real DAG -------------
  // One node per kernel-stage type within a single layer (11 nodes) -- the
  // smallest unit where every edge is a genuine measured dependency. This
  // replaces the old single "now executing" hero panel + scrolling ticker,
  // which both existed purely to answer "which of the 11 stages is active
  // right now" one at a time in text; the DAG answers that spatially, for
  // all 11 at once. The distinction that motivated this redesign: a
  // transformer forward pass is always a DAG (strictly forward, no cycles)
  // -- what repeats per token is an OUTER LOOP re-running this same DAG
  // shape, not a cycle inside it. That's encoded directly below: 10 solid
  // forward edges are the real DAG, and exactly one dashed violet loop-back
  // edge (down_proj -> input_layernorm, "×40 layers") represents the
  // repetition -- never drawn as just another forward edge.
  const snapshot = cuptiSnapshot ?? { connected: false, layerCount: 0, history: [], stages: [], queryStageCounts: [] };
  const dagStages = layerStages(QWEN3_14B);
  const dagOriginX = heroX + 34; // extra left margin so the loop-back arc has room to bulge without clipping
  const dagLabel = svgElement("text", { x: heroX, y: heroY - 8, class: "graph-footnote" });
  dagLabel.textContent = snapshot.connected
    ? "kernel execution · one transformer layer"
    : "connecting to CUPTI…";
  svg.append(dagLabel);

  const dagAnchors = dagStages.map((stage, index) => {
    const pos = dagNodePosition(index, dagOriginX, heroY);
    const stageState = snapshot.stages?.[index] ?? null;
    const entry = stageState?.entry ?? null;
    return addNode(svg, {
      x: pos.x, y: pos.y, width: DAG_NODE_W, height: DAG_NODE_H,
      color: entry ? "#72e3b1" : "#64c7e8",
      kind: entry ? "measured" : "illustrative",
      pulse: Boolean(stageState?.live),
      title: stage.title,
      titleLimit: 16,
      tooltip: entry
        ? `${stage.title} — ${stage.lines?.[0] ?? ""} · ${truncate(entry.name, 48)} · grid[${entry.grid.join(",")}]`
        : `${stage.title} — ${stage.lines?.[0] ?? ""} · no real launch observed yet this session`,
      onActivate: () => scrollToCard("kernel-activity-feed"),
    });
  });

  for (let i = 0; i < dagAnchors.length - 1; i += 1) {
    const toState = snapshot.stages?.[i + 1] ?? null;
    const bothMeasured = Boolean(snapshot.stages?.[i]?.entry) && Boolean(toState?.entry);
    connectDagStage(svg, dagAnchors[i], dagAnchors[i + 1], {
      color: "#72e3b1", marker: "arrow-measured", width: 1.4,
      kind: bothMeasured ? "measured" : "illustrative",
      flowing: liveActive && Boolean(toState?.live),
    });
  }

  // Loop-back edge: down_proj (last stage) -> input_layernorm (first stage),
  // drawn as a deliberate arc bulging left of both nodes rather than a
  // straight line -- the standard way flowcharts distinguish "repeat" from
  // a normal step, and the concrete visual answer to "is this a DAG or a
  // cycle" (it's a DAG; this one edge is explicitly the repetition, styled
  // differently on purpose).
  const loopFrom = dagAnchors[dagAnchors.length - 1];
  const loopTo = dagAnchors[0];
  const loopBendX = dagOriginX - 30;
  const loopMidY = (loopFrom.cy + loopTo.cy) / 2;
  const loopPath = svgElement("path", {
    d: `M ${loopFrom.left} ${loopFrom.cy} C ${loopBendX} ${loopFrom.cy}, ${loopBendX} ${loopTo.cy}, ${loopTo.left} ${loopTo.cy}`,
    class: "graph-edge matched dag-loop-edge",
    stroke: "#b89cff",
    "stroke-width": 1.6,
    "stroke-dasharray": "7 6",
    "marker-end": "url(#arrow-matched)",
  });
  svg.insertBefore(loopPath, svg.querySelector(".graph-node"));
  const loopLabel = svgElement("text", {
    x: loopBendX, y: loopMidY, class: "graph-edge-label matched", "text-anchor": "middle",
    transform: `rotate(-90 ${loopBendX} ${loopMidY})`,
  });
  loopLabel.textContent = "×40 layers";
  svg.append(loopLabel);

  // ---- Layer sweep: 40 real layer positions, this step ------------------
  // One cell per transformer layer. Lit (mint) once that layer's first real
  // launch has arrived this step; the currently-executing layer pulses.
  // This is the real per-layer signal the offline exporter can't show live
  // (no "total layers for the whole request" -- see the memory note on why
  // this resets every step): derived directly from cuptiSnapshot.layerLog,
  // never a paced/synthetic sweep.
  const layerLabel = svgElement("text", { x: heroX, y: layerY - 8, class: "graph-footnote" });
  layerLabel.textContent = "same 11 stages, repeated once per layer below · layer sweep this step";
  svg.append(layerLabel);

  const layerSweep = svgElement("g", { class: "layer-sweep" });
  svg.append(layerSweep);
  const layerLog = snapshot.layerLog ?? new Array(LAYER_COUNT).fill(null);
  layerLog.forEach((entry, index) => {
    const chipX = heroX + index * (LAYER_CHIP + LAYER_GAP);
    addStageChip(layerSweep, {
      x: chipX, y: layerY, size: LAYER_CHIP,
      color: entry ? "#72e3b1" : "#60756d",
      kind: entry ? "measured" : "illustrative",
      active: index === snapshot.layerIndex,
      label: "",
      title: entry
        ? `layer ${index}: ${entry.stageCount}/11 stages seen this step`
        : `layer ${index}: not reached yet this step`,
    });
  });

  // ---- Kernel call counts: real launches per stage, this query ----------
  // One bar per DAG stage -- how many times it has actually fired
  // (cupti-activity.js's queryStageCounts, real classified launches, zeroed
  // once per chat turn) since this response started, not this step. A
  // multi-token response makes every stage's real count large and roughly
  // even (layers x steps so far) except attn, which is the same layers x
  // steps count too since its 3 physical launches collapse into 1 stage
  // occurrence -- if the bars are visibly uneven, that itself is real
  // signal (a stage being skipped or double-counted), not decoration.
  const countsLabel = svgElement("text", { x: heroX, y: countsY - 8, class: "graph-footnote" });
  countsLabel.textContent = "kernel calls this query · real classified launches per stage";
  svg.append(countsLabel);

  const stageCounts = snapshot.queryStageCounts ?? new Array(dagStages.length).fill(0);
  const maxStageCount = Math.max(1, ...stageCounts);
  const countsBarX = heroX + COUNTS_LABEL_W;
  const countsBarW = (dagOriginX + DAG_TOP_WIDTH) - countsBarX - COUNTS_COUNT_W;
  dagStages.forEach((stage, index) => {
    const rowY = countsY + index * (COUNTS_ROW_H + COUNTS_ROW_GAP);
    const count = stageCounts[index] ?? 0;
    const label = svgElement("text", {
      x: heroX, y: rowY + COUNTS_ROW_H - 4, class: "graph-node-line",
    });
    label.textContent = truncate(stage.title, 18);
    svg.append(label);
    addProgressBar(svg, {
      x: countsBarX, y: rowY + 2, width: countsBarW, height: COUNTS_ROW_H - 6,
      fraction: count / maxStageCount, color: count > 0 ? "#72e3b1" : "#20332e",
    });
    const countText = svgElement("text", {
      x: countsBarX + countsBarW + COUNTS_COUNT_W - 4, y: rowY + COUNTS_ROW_H - 4,
      class: "graph-node-line", "text-anchor": "end",
    });
    countText.textContent = String(count);
    svg.append(countText);
  });

  // ---- Latency anatomy: real GPU-busy time over a rolling window --------
  // Deliberately NOT per-engine-step: CUPTI activity is delivered in
  // periodic bursts (cuptiActivityFlushPeriod), confirmed live -- most
  // ~120ms step windows receive zero events, then every 6-7 steps several
  // thousand arrive at once. Attributing that burst to whichever single
  // step happened to be in flight produced impossible numbers (700%+ on
  // burst steps, 0% on every other step) in initial testing.
  //
  // The window and the busy-time union are BOTH computed on CUPTI's own
  // device clock (see cupti-activity.js's getRollingBusy) -- an earlier
  // version measured the window in client arrival time instead, which
  // still overshot 100% (114-116%, confirmed live): bursty/delayed delivery
  // means the set of events arriving in a fixed *client-time* window can
  // correspond to device activity spanning a different, longer real span.
  // With one consistent clock for both, the merged union is mathematically
  // bounded by the window length -- this cannot exceed 100%, not just
  // empirically but by construction.
  const anatomyLabel = svgElement("text", { x: heroX, y: anatomyY - 8, class: "graph-footnote" });
  anatomyLabel.textContent = "latency anatomy · rolling GPU-busy window";
  svg.append(anatomyLabel);

  const anatomyWidth = LAYER_COUNT * (LAYER_CHIP + LAYER_GAP) - LAYER_GAP;
  const rolling = snapshot.rollingBusy ?? { busyNs: 0, windowNs: 0, launchCount: 0 };
  if (rolling.launchCount > 0) {
    const busyMs = rolling.busyNs / 1e6;
    const windowMs = rolling.windowNs / 1e6;
    const fraction = windowMs > 0 ? Math.min(1, busyMs / windowMs) : 0;
    addProgressBar(svg, {
      x: heroX, y: anatomyY, width: anatomyWidth, height: ANATOMY_BAR_HEIGHT,
      fraction, color: "#72e3b1",
    });
    const anatomyText = svgElement("text", {
      x: heroX, y: anatomyY + ANATOMY_BAR_HEIGHT + 16, class: "graph-node-line",
    });
    anatomyText.textContent = `${busyMs.toFixed(1)} ms GPU-busy of the last ${(windowMs / 1000).toFixed(1)}s of device activity (${Math.round(fraction * 100)}%) · ${rolling.launchCount} launches, CUPTI delivers in bursts so this is a rolling window on its own clock, not a single step`;
    svg.append(anatomyText);
  } else {
    const anatomyEmpty = svgElement("text", { x: heroX, y: anatomyY + 10, class: "graph-node-line" });
    anatomyEmpty.textContent = "waiting for the first real launch…";
    svg.append(anatomyEmpty);
  }

  const note = svgElement("text", { x: 18, y: graphHeight - 14, class: "graph-footnote" });
  note.textContent = snapshot.connected
    ? "Mint = a real CUPTI launch; the glow means it landed in the last 250ms. Ticker reads left (oldest) to right (now)."
    : "Connecting to the CUPTI stream…";
  svg.append(note);
}
