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

import { STAGE_TITLES } from "./kernel-graph.js?v=graph24";

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
};

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
  }
}

// Called by trace.js on every step_begin patch from the semantic-ring WS --
// reshape_and_cache fires exactly once per layer, so counting it since this
// step began is a real, measured "how far into the 40-layer sweep" signal.
export function resetCuptiStepCounter() {
  state.reshapeAndCacheCountThisStep = 0;
  state.currentLayerIndex = -1;
  state.layerLog = new Array(40).fill(null);
}

// Snapshot consumed by causal-graph.js's "now executing" hero + history
// ticker. `history` is the real chronological transition sequence (oldest
// first); `history[history.length - 1]` is the current/most-recent stage --
// the hero panel's subject. `stages` (per-stage-index most-recent launch)
// is kept too, in case a future view wants per-stage detail again.
export function getCuptiSnapshot() {
  const now = performance.now();
  return {
    connected: state.connected,
    layerCount: Math.min(state.reshapeAndCacheCountThisStep, 40),
    layerIndex: state.currentLayerIndex,
    layerLog: state.layerLog.map((entry) => entry
      ? { stageCount: entry.stagesSeen.size, live: now - entry.updatedAt < RECENCY_LIVE_MS }
      : null),
    stages: state.stages.map((entry, index) => ({
      index,
      title: STAGE_TITLES[index],
      entry,
      live: entry !== null && now - entry.updatedAt < RECENCY_LIVE_MS,
    })),
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
