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
} from "./graph-primitives.js?v=graph24";
import { STAGE_ABBREVIATIONS } from "./kernel-graph.js?v=graph24";

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
export function renderCausalGraph({ svg, trace, step, kernelIndex, onKernelSelect, liveActive = false, cuptiSnapshot = null }) {
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
    renderLiveGraph(svg, { trace, step, liveActive, cuptiSnapshot, scheduler });
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
const HERO_WIDTH = 640;
const HERO_HEIGHT = 118;
const TICKER_CHIP = 30;
const TICKER_GAP = 6;
const LAYER_CHIP = 12;
const LAYER_GAP = 3;
const LAYER_COUNT = 40;

function renderLiveGraph(svg, { trace, step, liveActive, cuptiSnapshot, scheduler }) {
  const packed = step?.packedSlices ?? [];
  const graphWidth = 18 + HERO_WIDTH + 24;
  const heroX = 18;
  const heroY = 76;
  const tickerY = heroY + HERO_HEIGHT + 20;
  const layerY = tickerY + TICKER_CHIP + 28;
  const graphHeight = layerY + LAYER_CHIP + 44;
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
  const tokenNode = addNode(svg, {
    x: 242, y: stripY, width: 130, height: stripH,
    color: "#64c7e8", kind: "measured", title: `${trace.query.tokens.length} tokens`,
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

  // ---- The main event: what is the GPU doing RIGHT NOW ------------------
  const snapshot = cuptiSnapshot ?? { connected: false, layerCount: 0, history: [] };
  const current = snapshot.history[snapshot.history.length - 1] ?? null;
  const heroMeasured = Boolean(current?.entry);

  const heroGroup = svgElement("g", {
    class: `graph-node ${heroMeasured ? "measured" : "illustrative"}${current?.live ? " live-active" : ""}`,
    transform: `translate(${heroX} ${heroY})`,
  });
  heroGroup.style.setProperty("--node-color", heroMeasured ? "#72e3b1" : "#64c7e8");
  heroGroup.append(svgElement("rect", { width: HERO_WIDTH, height: HERO_HEIGHT, rx: 16 }));

  const heroLabel = svgElement("text", { x: 24, y: 30, class: "graph-hero-label" });
  heroLabel.textContent = "NOW EXECUTING";
  heroGroup.append(heroLabel);

  const heroTitle = svgElement("text", { x: 24, y: 68, class: "graph-hero-title" });
  heroTitle.textContent = current ? current.title : (snapshot.connected ? "waiting for the next launch…" : "connecting to CUPTI…");
  heroGroup.append(heroTitle);

  const heroDetail = svgElement("text", { x: 24, y: 96, class: "graph-hero-detail" });
  heroDetail.textContent = current?.entry
    ? `${truncate(current.entry.name, 52)} · grid[${current.entry.grid.join(",")}]`
    : "no real launch observed yet this session";
  heroGroup.append(heroDetail);

  const heroBadge = svgElement("text", { x: HERO_WIDTH - 20, y: 30, class: "graph-hero-badge", "text-anchor": "end" });
  heroBadge.textContent = snapshot.connected ? `layer ${snapshot.layerCount}/40` : "disconnected";
  heroGroup.append(heroBadge);

  heroGroup.setAttribute("role", "button");
  heroGroup.setAttribute("tabindex", "0");
  heroGroup.classList.add("interactive");
  const goToFeed = () => scrollToCard("kernel-activity-feed");
  heroGroup.addEventListener("click", goToFeed);
  heroGroup.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") { event.preventDefault(); goToFeed(); }
  });
  svg.append(heroGroup);

  // ---- History ticker: the last several distinct stages, oldest to newest,
  // for just enough "what just happened" context without a diagram to read.
  const tickerLabel = svgElement("text", { x: heroX, y: tickerY - 8, class: "graph-footnote" });
  tickerLabel.textContent = "recent";
  svg.append(tickerLabel);

  if (!snapshot.history.length) {
    const empty = svgElement("text", { x: heroX, y: tickerY + 20, class: "graph-node-line" });
    empty.textContent = "waiting for the first real kernel launch…";
    svg.append(empty);
  } else {
    snapshot.history.forEach((entry, index) => {
      const chipX = heroX + index * (TICKER_CHIP + TICKER_GAP);
      const isCurrent = index === snapshot.history.length - 1;
      addStageChip(svg, {
        x: chipX, y: tickerY, size: TICKER_CHIP,
        color: entry.entry ? "#72e3b1" : "#64c7e8",
        kind: entry.entry ? "measured" : "illustrative",
        active: isCurrent && entry.live,
        label: STAGE_ABBREVIATIONS[entry.stageIndex],
        title: entry.title,
        onActivate: () => scrollToCard("kernel-activity-feed"),
      });
    });
  }

  // ---- Layer sweep: 40 real layer positions, this step ------------------
  // One cell per transformer layer. Lit (mint) once that layer's first real
  // launch has arrived this step; the currently-executing layer pulses.
  // This is the real per-layer signal the offline exporter can't show live
  // (no "total layers for the whole request" -- see the memory note on why
  // this resets every step): derived directly from cuptiSnapshot.layerLog,
  // never a paced/synthetic sweep.
  const layerLabel = svgElement("text", { x: heroX, y: layerY - 8, class: "graph-footnote" });
  layerLabel.textContent = "layer sweep · this step";
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

  const note = svgElement("text", { x: 18, y: graphHeight - 14, class: "graph-footnote" });
  note.textContent = snapshot.connected
    ? "Mint = a real CUPTI launch; the glow means it landed in the last 250ms. Ticker reads left (oldest) to right (now)."
    : "Connecting to the CUPTI stream…";
  svg.append(note);
}
