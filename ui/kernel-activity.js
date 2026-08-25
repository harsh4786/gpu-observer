// Real-time tail of the shadow container's (run-live-demo-shadow.sh)
// Compute Sanitizer device probe -- block-entry events for
// reshape_and_cache_flash_kernel, genuinely measured GPU kernel launches
// straight off the device (collector/src/bin/sanitizer_stream_server.rs).
// Runs on a SEPARATE container from the main causal graph's CUPTI data
// (cupti-activity.js): CUPTI and the sanitizer cannot share one process on
// this stack (single CUPTI subscriber slot), so this module also opens its
// own connection to the shadow's semantic ring (packed-row layout) to
// compute live per-event request attribution -- entirely self-contained,
// not driven by the primary container's step boundaries (which run on an
// independent, unsynchronized schedule; see run-live-demo-shadow.sh).
//
// Qwen3-14B emits exactly 40 block-entry events per decode step, one per
// transformer layer (EXP-0018), so events-this-step / 40 is a real, measured
// progress signal.
//
// Attribution formula (ported from export_trace_bundle.rs's kernel_view,
// see the plan): a block index b belongs to the packed-row owner whose
// [rowBegin, rowEnd) range contains it. The offline version also requires
// kernel.grid[1]==1 && kernel.grid[2]==1 as a validity precondition, checked
// against live CUPTI grid dims -- unavailable here (the shadow container
// runs the sanitizer, not CUPTI, and the sanitizer's own launch-geometry
// records are only populated in the crash-prone launch_identity mode, not
// the validated live config). This is carried over as an assumed stable
// structural fact about this kernel in this workload, not re-verified live.
//
// Only this one anchor kernel is patched (see the plan this was built
// from); there is nothing here to generalize to other kernels yet.

import { requestColor } from "./graph-primitives.js?v=graph36";

const LAYERS_PER_STEP = 40;
const RECENT_LIMIT = 8;
const RECONNECT_DELAY_MS = 2000;
// Real SM count for this box's GPU (NVIDIA GB10), confirmed via
// torch.cuda.get_device_properties(0).multi_processor_count, not guessed.
// The grid below is an SM-index grid for at-a-glance occupancy, not a
// physical floorplan -- this project has no verified physical SM topology
// to draw from, so it doesn't claim one.
const SM_COUNT = 48;

const byId = (id) => document.getElementById(id);

const state = {
  sanitizerWs: null,
  sanitizerReconnectTimer: null,
  semanticWs: null,
  semanticReconnectTimer: null,
  totalEvents: 0,
  stepEvents: 0,
  recent: [], // newest first, capped at RECENT_LIMIT; each entry may carry attributedRequestId
  packedRows: [], // this step's [{requestId, rowBegin, rowEnd}], from the shadow's own semantic ring
  smActivity: new Map(), // sm_id -> {count, updatedAt, kernel, color, ownerLabel}, real block-entry events only
  smCellEls: [], // 48 persistent DOM nodes, built once so a CSS blink animation can restart per event
};

function sanitizerWsUrl() {
  const params = new URLSearchParams(location.search);
  return params.get("san") ?? `ws://${location.hostname}:8190`;
}

function semanticWsUrl() {
  const params = new URLSearchParams(location.search);
  return params.get("shadowWs") ?? `ws://${location.hostname}:8189`;
}

function shortId(value) {
  const text = String(value ?? "");
  return text.length > 12 ? `${text.slice(0, 6)}…${text.slice(-4)}` : text;
}

// Ported from export_trace_bundle.rs's kernel_view ownership loop, minus the
// grid[0] clamp (a batch-computation concern; not needed for a single
// observed block index) -- see the module header comment.
function attributeBlock(blockX) {
  const owner = state.packedRows.find((row) => blockX >= row.rowBegin && blockX < row.rowEnd);
  return owner ?? null;
}

function render() {
  const progress = Math.min(state.stepEvents, LAYERS_PER_STEP);
  byId("ka-step-progress").textContent = `${progress}/${LAYERS_PER_STEP}`;
  byId("ka-step-count").textContent = String(state.stepEvents);
  byId("ka-total-count").textContent = String(state.totalEvents);
  byId("ka-progress-fill").style.width = `${(progress / LAYERS_PER_STEP) * 100}%`;

  const feed = byId("kernel-activity-feed");
  if (!state.recent.length) {
    feed.innerHTML = '<div class="empty-state">Waiting for the next decode step’s kernel launches…</div>';
    return;
  }
  feed.innerHTML = state.recent.map((event) => {
    const block = Array.isArray(event.block) ? event.block.map(Number).join(", ") : "?";
    const attribution = event.owner
      ? `<span class="ka-event-owner">${shortId(event.owner.requestId)}</span>`
      : `<span class="ka-event-owner unattributed">unattributed</span>`;
    return `<div class="ka-event"><span class="ka-event-kernel">${event.kernel}</span>${attribution}<span>SM ${Number(event.sm_id)}</span><span>block [${block}]</span></div>`;
  }).join("");
}

function handleSanitizerEvent(event) {
  state.totalEvents += 1;
  state.stepEvents += 1;
  const blockX = Number(event.block?.[0]);
  const owner = Number.isFinite(blockX) ? attributeBlock(blockX) : null;
  state.recent.unshift({ ...event, owner });
  if (state.recent.length > RECENT_LIMIT) state.recent.length = RECENT_LIMIT;

  const smId = Number(event.sm_id);
  if (Number.isInteger(smId) && smId >= 0 && smId < SM_COUNT) {
    const prior = state.smActivity.get(smId);
    // This kernel fires exactly once per layer (module header comment), so
    // stepEvents -- already reset to 0 on every step_begin from the
    // shadow's own semantic ring -- directly IS the 1-indexed count of
    // layers seen so far this step. No separate layer tracker needed; this
    // is the shadow container's OWN decode progress, independent of the
    // primary container's (unsynchronized schedules, see module header).
    const layerIndex = Math.min(state.stepEvents - 1, LAYERS_PER_STEP - 1);
    const activity = {
      count: (prior?.count ?? 0) + 1,
      updatedAt: performance.now(),
      kernel: event.kernel,
      layerIndex,
      // Same request-color palette causal-graph.js's lanes use, so a blink
      // reads as "this request, on this SM, right now" -- real attribution
      // data that was already computed for the event feed and just wasn't
      // reaching the grid before.
      color: owner ? requestColor(owner.requestId) : "#72e3b1",
      ownerLabel: owner ? shortId(owner.requestId) : "unattributed",
    };
    state.smActivity.set(smId, activity);
    ensureSmGrid();
    blinkSm(smId, activity);
    updateSmSubtitle(event.kernel);
  }

  render();
}

// Built once (not rebuilt per-event) so a CSS animation can be restarted on
// an already-mounted node -- rebuilding the DOM every event would mean a
// repeat hit on the same SM (very common: this kernel launches one block
// per call and the GPU scheduler often reuses the same SM for stretches)
// never gets a fresh blink, just a static color.
function ensureSmGrid() {
  if (state.smCellEls.length) return;
  const grid = byId("sm-grid");
  if (!grid) return;
  const frag = document.createDocumentFragment();
  for (let sm = 0; sm < SM_COUNT; sm += 1) {
    const cell = document.createElement("div");
    cell.className = "sm-cell";
    cell.textContent = String(sm);
    cell.title = `SM ${sm}: no event yet`;
    cell.addEventListener("animationend", () => cell.classList.remove("blink"));
    frag.append(cell);
    state.smCellEls.push(cell);
  }
  grid.append(frag);
}

// Real per-launch blink: every block-entry event flashes its SM's cell,
// restarting the animation even for back-to-back hits on the same SM (the
// classList remove + forced reflow + re-add is the standard trick for
// retriggering a CSS animation on a node that already has it).
function blinkSm(smId, activity) {
  const cell = state.smCellEls[smId];
  if (!cell) return;
  cell.classList.add("touched");
  cell.style.setProperty("--sm-color", activity.color);
  cell.title = `SM ${smId}: layer ${activity.layerIndex}/${LAYERS_PER_STEP} · ${activity.count} block-entry event${activity.count === 1 ? "" : "s"} · last owner: ${activity.ownerLabel}`;
  cell.classList.remove("blink");
  void cell.offsetWidth;
  cell.classList.add("blink");
}

function updateSmSubtitle(kernel) {
  const subtitle = byId("sm-grid-subtitle");
  if (!subtitle) return;
  subtitle.textContent = `${kernel} · ${state.smActivity.size}/${SM_COUNT} SMs touched this session · 1 block/launch during decode, so ≤1 blinking at a time is real, not stuck`;
}

function handleQuality(patch) {
  const status = byId("kernel-activity-status");
  const drops = Number(patch.drops) || 0;
  const errors = Number(patch.sequence_errors) || 0;
  if (drops > 0 || errors > 0) {
    status.title = `${drops} ring drops, ${errors} sequence errors since capture start (of ${patch.total_forwarded} forwarded).`;
  }
}

function handleSemanticPatch(patch) {
  if (patch.kind === "step_begin") {
    state.stepEvents = 0;
    state.packedRows = [];
    render();
  } else if (patch.kind === "packed_request_slice") {
    state.packedRows.push({
      requestId: patch.request,
      rowBegin: Number(patch.row_begin),
      rowEnd: Number(patch.row_end),
    });
  }
}

export function connectKernelActivity() {
  const status = byId("kernel-activity-status");
  let socket;
  try {
    socket = new WebSocket(sanitizerWsUrl());
  } catch (error) {
    status.textContent = "disconnected";
    status.className = "status disconnected";
    scheduleSanitizerReconnect();
    return;
  }
  state.sanitizerWs = socket;
  socket.addEventListener("open", () => {
    status.textContent = "connected";
    status.className = "status connected";
  });
  socket.addEventListener("message", (raw) => {
    let patch;
    try {
      patch = JSON.parse(raw.data);
    } catch (error) {
      return; // a malformed frame must never take down the live feed
    }
    if (patch.kind === "kernel_block_event") handleSanitizerEvent(patch);
    else if (patch.kind === "sanitizer_quality") handleQuality(patch);
  });
  socket.addEventListener("close", () => {
    status.textContent = "disconnected";
    status.className = "status disconnected";
    scheduleSanitizerReconnect();
  });
  socket.addEventListener("error", () => socket.close());

  connectShadowSemantic();
  ensureSmGrid(); // paint all 48 (unlit) cells immediately, don't wait for the first event
}

function connectShadowSemantic() {
  let socket;
  try {
    socket = new WebSocket(semanticWsUrl());
  } catch (error) {
    scheduleSemanticReconnect();
    return;
  }
  state.semanticWs = socket;
  socket.addEventListener("message", (raw) => {
    let patch;
    try {
      patch = JSON.parse(raw.data);
    } catch (error) {
      return;
    }
    handleSemanticPatch(patch);
  });
  socket.addEventListener("close", () => scheduleSemanticReconnect());
  socket.addEventListener("error", () => socket.close());
}

function scheduleSanitizerReconnect() {
  if (state.sanitizerReconnectTimer) return;
  state.sanitizerReconnectTimer = setTimeout(() => {
    state.sanitizerReconnectTimer = null;
    connectKernelActivity();
  }, RECONNECT_DELAY_MS);
}

function scheduleSemanticReconnect() {
  if (state.semanticReconnectTimer) return;
  state.semanticReconnectTimer = setTimeout(() => {
    state.semanticReconnectTimer = null;
    connectShadowSemantic();
  }, RECONNECT_DELAY_MS);
}

render();
