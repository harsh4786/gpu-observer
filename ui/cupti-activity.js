// Live WS client for cupti_activity_agent's real per-kernel-launch stream
// (collector/src/bin/cupti_stream_server.rs, port 8090 by default) --
// classifies each launch into one of Qwen3-14B's 11 per-layer stages
// (kernel-graph.js STAGE_TITLES) and keeps the most recent real launch per
// stage. This replaces the old illustrative sweep entirely: with real
// per-launch data arriving continuously during decode, "which stage most
// recently received a launch" IS the honest live-activity signal -- no
// synthetic pacing needed.
//
// Classifier basis (empirically captured this session, not guessed --
// see the plan's Part 0): one transformer layer emits a fixed 13-launch
// cycle. Four stages (qkv_proj, o_proj, gate_up_proj, down_proj) share the
// same kernel symbol during decode (cuBLAS gemvx::kernel) and during
// prefill (cutlass/nvjet GEMMs) -- name alone can't tell them apart. They're
// disambiguated by grid_x where it's unique (qkv_proj=1792, matching GQA
// 40:8 head math: 7168/4; gate_up_proj=8704, matching 2*intermediate_size/4)
// and otherwise by launch position (o_proj immediately follows the attn
// group; down_proj immediately follows SiluAndMul) -- both share grid_x=1280
// since both output hidden_size=5120. attn itself is 3 physical launches
// (flash_fwd_splitkv[_combine] + a memcpy) treated as one stage until a
// non-attn-family name breaks the run.
//
// Real captured cycle, in architectural order:
//   input_layernorm -> qkv_proj -> q_norm/k_norm -> rotary_emb ->
//   reshape_and_cache -> attn -> o_proj -> post_attention_layernorm ->
//   gate_up_proj -> SiluAndMul -> down_proj -> (next layer's input_layernorm)

import { STAGE_TITLES } from "./kernel-graph.js?v=graph48";

const [
  INPUT_LAYERNORM, QKV_PROJ, QK_NORM, ROTARY_EMB, RESHAPE_AND_CACHE,
  ATTN, O_PROJ, POST_ATTN_LAYERNORM, GATE_UP_PROJ, SILU_AND_MUL, DOWN_PROJ,
] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

const RECENCY_LIVE_MS = 250; // a chip counts as "currently active" if updated more recently than this
const HISTORY_LIMIT = 14; // how many distinct recent stage transitions the ticker keeps

const state = {
  ws: null,
  reconnectTimer: null,
  connected: false,
  // Per-stage-index most-recent real launch: {name, grid, block, correlationId, startNs, updatedAt}
  stages: new Array(STAGE_TITLES.length).fill(null),
  // Chronological sequence of distinct stage transitions -- {stageIndex, at}, oldest
  // first, capped at HISTORY_LIMIT. Consecutive launches classified into the SAME
  // stage (e.g. attn's 3 physical kernels) collapse into one entry so the ticker
  // reads as a clean transition log, not a flood of repeats.
  recentSequence: [],
  // Real classified-launch tally per stage, scoped to the current query --
  // zeroed once per chat turn (resetCuptiQueryCounters, called from
  // trace.js's sendChatMessage), not per step. Answers "how many times was
  // this kernel-stage actually called for this prompt", e.g. attn/qkv_proj/
  // etc. firing once per layer per step, accumulated across every step of
  // the response so far -- split into thinking vs response buckets by
  // inThinkingPhase below.
  queryStageCounts: new Array(STAGE_TITLES.length).fill(null).map(() => ({ thinking: 0, response: 0 })),
  // Set by trace.js's setThinkingPhase(), driven by watching the streamed
  // SSE reply text for Qwen3's own <think>/</think> tags -- a real content
  // boundary, but arriving over a completely separate channel (the HTTP
  // chat stream) from the kernel-launch events counted below (the CUPTI
  // WS). So a launch's thinking/response bucket is a reconstructed
  // correlation of two independent real streams, accurate to roughly
  // "whichever phase was current when this launch was processed" -- not
  // a measured per-token attribution. Defaults false (response) so output
  // with no thinking block at all is never miscategorized as thinking.
  inThinkingPhase: false,
  reshapeAndCacheCountThisStep: 0,
  lastRecognizedStage: null, // for disambiguating the two grid_x=1280 gemvx cases
  // Per-layer retention for this step: index 0..39, null until that layer's
  // first real launch arrives. A new layer always starts with an
  // INPUT_LAYERNORM classification (rsqrt_2 for layers 1-39, the fused
  // embedding_rms_norm for layer 0) -- real, not inferred -- so that's what
  // advances currentLayerIndex. stagesSeen is a Set of stage indices, so
  // "how much of this layer's 11-stage cycle has real launch evidence so
  // far this step" is a real, measured count, not a guess.
  currentLayerIndex: -1,
  layerLog: new Array(40).fill(null),
  // Rolling window of real launches -- {startNs, endNs}, both CUPTI's own
  // device clock. Deliberately NOT bucketed per engine step:
  // cuptiActivityFlushPeriod delivers activity in periodic bursts (confirmed
  // live -- most ~120ms step windows get zero events, then every 6-7 steps
  // several thousand arrive at once), so "launches that arrived during step
  // N's WS window" massively misattributes -- most steps read 0%, the step
  // that catches a burst reads 700%+.
  //
  // The window boundary is also computed on CUPTI's own clock (via
  // latestEndNs below), NOT client arrival time (performance.now()) -- an
  // earlier version used arrival time for the window and device time for
  // durations, which still produced >100% readings (114-116%, confirmed
  // live): because delivery is bursty and delayed, the set of events
  // arriving in any fixed *client-time* window can correspond to device
  // activity spanning a *different*, sometimes longer, real span. Defining
  // the window in the interval data's own clock makes the merged union
  // mathematically bounded by the window length -- it cannot exceed 100%.
  recentLaunches: [],
  latestEndNs: 0,
  firstStartNs: null, // set once, on the very first real launch -- caps windowNs early in a session
};

const ROLLING_WINDOW_NS = 3_000_000_000; // 3s, on CUPTI's device clock

function mergedBusyNs(intervals) {
  if (!intervals.length) return 0;
  const sorted = [...intervals].sort((a, b) => a[0] - b[0]);
  let busy = 0;
  let [curStart, curEnd] = sorted[0];
  for (let i = 1; i < sorted.length; i += 1) {
    const [start, end] = sorted[i];
    if (start <= curEnd) {
      curEnd = Math.max(curEnd, end);
    } else {
      busy += curEnd - curStart;
      [curStart, curEnd] = [start, end];
    }
  }
  busy += curEnd - curStart;
  return busy;
}

function isGemmFamily(name) {
  return name.includes("gemvx") || name.includes("cutlass") || name.includes("nvjet");
}

function isAttnFamily(name) {
  return name.includes("flash_fwd");
}

function classify(event) {
  const name = event.name;
  const gridX = Number(event.grid?.[0] ?? 0);

  if (name.includes("reshape_and_cache_flash_kernel")) {
    state.reshapeAndCacheCountThisStep += 1;
    return RESHAPE_AND_CACHE;
  }
  if (isAttnFamily(name)) return ATTN;
  if (name === "memcpy32_post" && state.lastRecognizedStage === ATTN) return ATTN;
  if (name.includes("mul_silu")) return SILU_AND_MUL;
  if (name.includes("rsqrt_0")) return POST_ATTN_LAYERNORM;
  if (name.includes("rsqrt_2")) return INPUT_LAYERNORM;
  if (name.includes("embedding_rms_norm")) return INPUT_LAYERNORM; // layer-0 special case, fused with embedding lookup
  if (name.includes("triton_red_fused_3")) return QK_NORM;
  if (name.includes("triton_poi_fused_4")) return ROTARY_EMB;
  if (isGemmFamily(name)) {
    if (gridX === 1792) return QKV_PROJ;
    if (gridX === 8704) return GATE_UP_PROJ;
    // Ambiguous grid (1280 during decode, or any prefill GEMM shape): resolve
    // by what stage was last confidently recognized -- o_proj follows attn,
    // down_proj follows SiluAndMul. If neither just happened, don't guess.
    if (state.lastRecognizedStage === ATTN) return O_PROJ;
    if (state.lastRecognizedStage === SILU_AND_MUL) return DOWN_PROJ;
    return null;
  }
  return null; // sampling/housekeeping kernels (radix sort, slot mapping, etc.) -- not one of the 11 stages
}

function handleEvent(event) {
  // Real GPU-busy accounting counts every launch, classified or not --
  // sampling/housekeeping kernels are still real GPU time, and excluding
  // them would understate busy time. Kept separate from the 11-stage
  // classifier below, which only cares about named pipeline stages.
  const startNs = Number(event.startNs);
  const endNs = Number(event.endNs);
  if (Number.isFinite(startNs) && Number.isFinite(endNs) && endNs > startNs) {
    state.recentLaunches.push({ startNs, endNs });
    if (endNs > state.latestEndNs) state.latestEndNs = endNs;
    if (state.firstStartNs === null) state.firstStartNs = startNs;
  }

  const stageIndex = classify(event);
  if (stageIndex === null) return;
  state.lastRecognizedStage = stageIndex;
  const now = performance.now();
  state.stages[stageIndex] = {
    name: event.name,
    grid: event.grid,
    block: event.block,
    correlationId: event.correlationId,
    startNs: event.startNs,
    updatedAt: now,
  };

  if (stageIndex === INPUT_LAYERNORM) state.currentLayerIndex += 1;
  if (state.currentLayerIndex >= 0 && state.currentLayerIndex < 40) {
    const layer = state.layerLog[state.currentLayerIndex]
      ?? (state.layerLog[state.currentLayerIndex] = { stagesSeen: new Set(), updatedAt: now });
    layer.stagesSeen.add(stageIndex);
    layer.updatedAt = now;
  }

  const last = state.recentSequence[state.recentSequence.length - 1];
  if (last && last.stageIndex === stageIndex) {
    last.at = now; // still the same stage as last transition: refresh recency, don't duplicate
  } else {
    state.recentSequence.push({ stageIndex, at: now });
    if (state.recentSequence.length > HISTORY_LIMIT) state.recentSequence.shift();
    // One distinct stage occurrence, same unit the DAG node represents --
    // attn's 3 physical launches (flash_fwd_splitkv[_combine] + memcpy) only
    // land here once, on the transition in, matching the "attn ran once
    // this layer" the DAG shows rather than triple-counting it.
    state.queryStageCounts[stageIndex][state.inThinkingPhase ? "thinking" : "response"] += 1;
  }
}

// Called by trace.js once per new chat turn (sendChatMessage), not per
// step -- these are per-query totals, so they should only zero when the
// query itself changes.
export function resetCuptiQueryCounters() {
  state.queryStageCounts = new Array(STAGE_TITLES.length).fill(null).map(() => ({ thinking: 0, response: 0 }));
  state.inThinkingPhase = false;
}

// Called by trace.js's sendChatMessage on every streamed SSE chunk, from
// watching the reply text itself for <think>/</think> -- see the state.
// inThinkingPhase comment above for why this is a reconstructed boundary,
// not a measured one. Idempotent (just an assignment), so it's safe to call
// on every chunk rather than needing a "did we already flip this" guard.
export function setThinkingPhase(isThinking) {
  state.inThinkingPhase = isThinking;
}

// Called by trace.js on every step_begin patch from the semantic-ring WS --
// reshape_and_cache fires exactly once per layer, so counting it since this
// step began is a real, measured "how far into the 40-layer sweep" signal.
export function resetCuptiStepCounter() {
  state.reshapeAndCacheCountThisStep = 0;
  state.currentLayerIndex = -1;
  state.layerLog = new Array(40).fill(null);
  // recentLaunches is deliberately NOT reset here -- it's a rolling window,
  // pruned by age (see getRollingBusy), independent of step boundaries.
}

// Real GPU-busy time over the last ROLLING_WINDOW_NS, entirely on CUPTI's
// own device clock (see the state comment for why arrival time can't be
// used for the window boundary). Prunes the rolling buffer as a side
// effect. Because both the window and the intervals live on one clock, the
// merged union is mathematically bounded by the window length -- fraction
// can never exceed 1 by construction, not just by an empirical cap.
export function getRollingBusy() {
  if (state.latestEndNs === 0) return { busyNs: 0, windowNs: 0, launchCount: 0 };
  const cutoff = Math.max(state.latestEndNs - ROLLING_WINDOW_NS, state.firstStartNs);
  state.recentLaunches = state.recentLaunches.filter((l) => l.endNs >= cutoff);
  const windowNs = state.latestEndNs - cutoff; // < ROLLING_WINDOW_NS early in a session, then pinned to it
  return {
    busyNs: mergedBusyNs(state.recentLaunches.map((l) => [Math.max(l.startNs, cutoff), l.endNs])),
    windowNs,
    launchCount: state.recentLaunches.length,
  };
}

// Snapshot consumed by causal-graph.js's "now executing" hero + history
// ticker. `history` is the real chronological transition sequence (oldest
// first); `history[history.length - 1]` is the current/most-recent stage --
// the hero panel's subject. `stages` (per-stage-index most-recent launch)
// is kept too, in case a future view wants per-stage detail again.
export function getCuptiSnapshot() {
  const now = performance.now();
  // The single most-recently-processed stage, same source state.
  // currentLayerIndex already uses for the layer sweep below. Needed
  // because CUPTI delivers in bursts (module header): a naive per-stage
  // "updated within the last 250ms" test goes true for most/all 11 stages
  // at once during a burst, so the whole DAG pulsed together instead of
  // tracking the one stage that actually just ran.
  const latest = state.recentSequence[state.recentSequence.length - 1] ?? null;
  return {
    connected: state.connected,
    rollingBusy: getRollingBusy(),
    layerCount: Math.min(state.reshapeAndCacheCountThisStep, 40),
    layerIndex: state.currentLayerIndex,
    layerLog: state.layerLog.map((entry) => entry
      ? { stageCount: entry.stagesSeen.size, live: now - entry.updatedAt < RECENCY_LIVE_MS }
      : null),
    stages: state.stages.map((entry, index) => ({
      index,
      title: STAGE_TITLES[index],
      entry,
      live: latest !== null && index === latest.stageIndex && now - latest.at < RECENCY_LIVE_MS,
    })),
    queryStageCounts: state.queryStageCounts.map((counts) => ({ ...counts })),
    history: state.recentSequence.map(({ stageIndex, at }) => ({
      stageIndex,
      title: STAGE_TITLES[stageIndex],
      entry: state.stages[stageIndex],
      live: now - at < RECENCY_LIVE_MS,
    })),
  };
}

function wsUrl() {
  const params = new URLSearchParams(location.search);
  return params.get("cupti") ?? `ws://${location.hostname}:8090`;
}

export function connectCuptiActivity() {
  let socket;
  try {
    socket = new WebSocket(wsUrl());
  } catch (error) {
    state.connected = false;
    scheduleReconnect();
    return;
  }
  state.ws = socket;
  socket.addEventListener("open", () => {
    state.connected = true;
  });
  socket.addEventListener("message", (raw) => {
    let event;
    try {
      event = JSON.parse(raw.data);
    } catch (error) {
      return; // a malformed frame must never take down the live feed
    }
    if (event.kind === "kernel_launch") handleEvent(event);
  });
  socket.addEventListener("close", () => {
    state.connected = false;
    scheduleReconnect();
  });
  socket.addEventListener("error", () => socket.close());
}

function scheduleReconnect() {
  if (state.reconnectTimer) return;
  state.reconnectTimer = setTimeout(() => {
    state.reconnectTimer = null;
    connectCuptiActivity();
  }, 2000);
}
