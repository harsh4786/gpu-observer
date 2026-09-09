import { renderCausalGraph } from "./causal-graph.js?v=graph48";
import { connectKernelActivity } from "./kernel-activity.js?v=graph48";
import { connectCuptiActivity, getCuptiSnapshot, resetCuptiStepCounter, resetCuptiQueryCounters, setThinkingPhase } from "./cupti-activity.js?v=graph48";
const GPU_REFRESH_INTERVAL_MS = 150; // re-render cadence for freshly arrived real CUPTI data, not a paced sweep

const state = {
  trace: null,
  stepIndex: 0,
  kernelIndex: 0,
  // Live-mode fields. Unused (stay at defaults) when viewing a sealed
  // TraceBundle file -- offline viewing behaves exactly as before.
  liveMode: false,
  liveStepIds: new Map(), // step-id string -> index into trace.steps, for O(1) patch application
  liveFocusHash: null, // which request's hash this chat turn is about; see resolveLiveFocus()
  liveFocusPromptTokens: null, // real prompt token count, captured off this turn's first request_slice patch -- see resolveLiveFocus()
  pendingFocusReset: false,
  liveStepActive: false,
  ws: null,
  wsReconnectTimer: null,
  chatBusy: false,
  activeAbort: null, // AbortController for the in-flight chat request, if any -- see the Stop button
};
const byId = (id) => document.getElementById(id);
const COLORS = ["#72e3b1", "#64c7e8", "#ffc66d", "#b89cff", "#ff8b7a", "#77a7ff"];

function escapeMarkup(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function shortId(value) {
  const text = String(value ?? "");
  return text.length > 12 ? `${text.slice(0, 6)}…${text.slice(-4)}` : text;
}

function requestColor(requestId) {
  let hash = 0;
  for (const char of String(requestId)) hash = (hash * 33 + char.charCodeAt(0)) >>> 0;
  return COLORS[hash % COLORS.length];
}

function validateTrace(trace) {
  if (!trace || trace.schema !== "GPU_OBSERVER_TRACE_V2") {
    throw new Error("Expected schema GPU_OBSERVER_TRACE_V2.");
  }
  if (!trace.query || !trace.focus || !Array.isArray(trace.steps) || !trace.steps.length) {
    throw new Error("TraceBundle is missing query, focus, or engine steps.");
  }
  if (!Array.isArray(trace.query.tokens) || !trace.evidence
      || !Array.isArray(trace.evidence.measured)
      || !Array.isArray(trace.evidence.reconstructed)
      || !Array.isArray(trace.evidence.unavailable)) {
    throw new Error("TraceBundle is missing token or evidence arrays.");
  }
  if (Boolean(trace.deepReplay) !== Boolean(trace.evidence.matchedReplay)) {
    throw new Error("TraceBundle deep replay and evidence status disagree.");
  }
  if (trace.deepReplay) {
    if (trace.deepReplay.schema !== "GPU_OBSERVER_DEEP_REPLAY_01"
        || trace.deepReplay.replayFingerprint !== trace.run?.replayFingerprint
        || trace.deepReplay.semanticSignature !== trace.run?.semanticSignature
        || !Array.isArray(trace.deepReplay.functions)
        || !trace.deepReplay.functions.length
        || !Array.isArray(trace.evidence.matched)) {
      throw new Error("Deep replay is missing or does not match the timed trace.");
    }
  }
  for (const step of trace.steps) {
    for (const field of ["schedulerSlices", "packedSlices", "focusedPackedTokens", "acceptedOutputTokens", "kernels"]) {
      if (!Array.isArray(step[field])) throw new Error(`Step ${step.id} is missing ${field}.`);
    }
  }
  return trace;
}

function metric(label, value) {
  return `<div class="metric"><strong>${escapeMarkup(value)}</strong><small>${escapeMarkup(label)}</small></div>`;
}

function currentStep() {
  return state.trace.steps[state.stepIndex];
}

function currentKernel() {
  return currentStep()?.kernels[state.kernelIndex] ?? null;
}

function renderGraph() {
  const step = currentStep();
  const liveActive = state.liveMode && state.liveStepActive;
  renderCausalGraph({
    svg: byId("causal-graph"),
    trace: state.trace,
    step,
    kernelIndex: state.kernelIndex,
    onKernelSelect: selectKernel,
    liveActive,
    // Gated on liveMode, NOT liveActive: liveActive goes false on every
    // step_end (including the final one when decoding stops), and gating
    // the whole snapshot on it wiped real cumulative data -- the per-query
    // kernel-call counts, the DAG's last-seen entries -- back to nothing
    // right when the response finished. liveActive still separately gates
    // the "flowing"/pulse animation (see renderLiveGraph) -- only whether
    // real accumulated data is SHOWN AT ALL should track liveMode.
    cuptiSnapshot: state.liveMode ? getCuptiSnapshot() : null,
    promptTokenCount: state.liveFocusPromptTokens,
  });
}

function selectStep(index) {
  state.stepIndex = Math.max(0, Math.min(index, state.trace.steps.length - 1));
  state.kernelIndex = 0;
  renderStep();
}

function selectKernel(index) {
  state.kernelIndex = Math.max(0, Math.min(index, currentStep().kernels.length - 1));
  renderKernelDetail();
  renderSass();
  renderGraph();
}

function renderHeader() {
  const run = state.trace.run ?? {};
  const fixture = String(run.captureStatus ?? "").includes("fixture");
  byId("run-subtitle").textContent = [
    run.model ?? "unknown model",
    run.executionMode ?? "unknown execution mode",
    `${state.trace.metrics?.engineSteps ?? state.trace.steps.length} engine steps`,
    `${state.trace.metrics?.cuptiKernels ?? 0} CUPTI kernels`,
  ].join(" · ");
  const status = byId("capture-status");
  if (state.liveMode) {
    status.textContent = "live capture — engine-step semantics, no CUPTI yet";
    status.className = "status illustrative";
  } else {
    status.textContent = fixture ? "schema fixture — not evidence" : "sealed measured trace";
    status.className = fixture ? "status fixture" : "status";
  }
}

function renderStepControls() {
  const focus = state.trace.focus.requestHash;
  const preferred = state.trace.steps.findIndex((step) =>
    step.focusedPackedTokens.some((token) => token.requestId === focus)
      && step.kernels.length);
  if (preferred >= 0 && !state.trace.steps[state.stepIndex]?.kernels.length) state.stepIndex = preferred;

  const select = byId("graph-step-select");
  select.innerHTML = state.trace.steps.map((step, index) =>
    `<option value="${index}">Step ${escapeMarkup(step.id)} · ${escapeMarkup(step.phase)} · ${step.scheduledTokens} tokens</option>`
  ).join("");
  select.value = String(state.stepIndex);
  select.onchange = () => selectStep(Number(select.value));
}

function renderStep() {
  renderStepControls();
  const step = currentStep();
  if (!step) {
    // Live mode between "send" and the first WS step_begin patch: no engine
    // step exists yet. Show that honestly instead of crashing on a null
    // step. The causal graph itself still renders (query/tokenizer lanes
    // form immediately -- see renderCausalGraph's null-step handling).
    byId("kernel-detail").innerHTML = '<div class="empty-state">No engine step yet.</div>';
    byId("ownership-list").innerHTML = "";
    renderGraph();
    return;
  }
  renderKernelDetail();
  renderSass();
  renderGraph();
}

function renderKernelDetail() {
  const kernel = currentKernel();
  const detail = byId("kernel-detail");
  const owners = byId("ownership-list");
  owners.innerHTML = "";
  if (!kernel) {
    detail.innerHTML = '<div class="empty-state">Select a step with CUPTI kernels.</div>';
    return;
  }
  detail.innerHTML = `
    <h3>${escapeMarkup(kernel.name)}</h3>
    <div class="detail-grid">
      ${metric("GPU duration", `${kernel.durationUs.toFixed(3)} µs`)}
      ${metric("stream", kernel.stream)}
      ${metric("correlation", kernel.correlationId)}
      ${metric("grid", kernel.grid.join(" × "))}
      ${metric("block", kernel.block.join(" × "))}
      ${metric("graph node", kernel.graphNodeId || "ordinary")}
    </div>
    <p class="why"><span class="badge ${kernel.attribution.startsWith("validated") ? "reconstructed" : "unknown"}">${escapeMarkup(kernel.attribution)}</span>
    Host submit ${kernel.runtimeSubmitFromStepUs.toFixed(2)} µs, GPU start ${kernel.startFromStepUs.toFixed(2)} µs from the same EngineCore step start.</p>
  `;
  if (!kernel.requestBlockOwnership.length) {
    owners.innerHTML = '<div class="empty-state">This kernel family has only honest step-level many-to-many attribution.</div>';
    return;
  }
  const gridX = Math.max(1, kernel.grid[0]);
  kernel.requestBlockOwnership.forEach((owner) => {
    const row = document.createElement("div");
    row.className = "owner-row";
    const width = ((owner.blockEnd - owner.blockBegin) / gridX) * 100;
    const left = (owner.blockBegin / gridX) * 100;
    row.innerHTML = `<span>${escapeMarkup(shortId(owner.requestId))}</span>
      <div class="owner-track"><div class="owner-fill" style="margin-left:${left}%;width:${width}%;--owner-color:${requestColor(owner.requestId)}"></div></div>
      <span>blocks [${owner.blockBegin},${owner.blockEnd})</span>`;
    owners.append(row);
  });
}

function renderSass() {
  const deep = state.trace.deepReplay;
  const kernel = currentKernel();
  const status = byId("sass-status");
  const content = byId("sass-content");
  if (!deep || !kernel) {
    status.className = "badge unknown";
    status.textContent = "not captured";
    content.innerHTML = '<div class="empty-state">Run the matched Compute Sanitizer replay to add sampled PC events and function PC ranges. The CUPTI timed run remains the latency source of truth.</div>';
    return;
  }
  const fn = deep.functions?.find((entry) =>
    entry.name === kernel.name || (entry.match && kernel.name.includes(entry.match)));
  if (!fn) {
    status.className = "badge unknown";
    status.textContent = "no matching function";
    content.innerHTML = '<div class="empty-state">The diagnostic replay has no function matching this selected kernel.</div>';
    return;
  }
  status.className = "badge matched";
  status.textContent = "matched replay";
  const sass = fn.sass ?? [];
  content.innerHTML = `
    <div class="sass-meta">fingerprint ${escapeMarkup(deep.replayFingerprint)} · module ${escapeMarkup(deep.moduleHash ?? fn.moduleHash ?? "unknown")} · function PC ${escapeMarkup(fn.functionPc ?? "unknown")} · size ${escapeMarkup(fn.functionSize ?? "unknown")}</div>
    <div class="sass-table">${sass.map((row) =>
      `<div class="sass-row${row.samples ? " hit" : ""}"><span class="offset">${escapeMarkup(row.offset)}</span><span>${escapeMarkup(row.instruction)}</span><span class="samples">${row.samples ?? 0} hits</span></div>`
    ).join("") || '<div class="empty-state">Function matched, but no disassembly rows were bundled.</div>'}</div>
  `;
}

function renderEvidence() {
  const evidence = state.trace.evidence;
  const columns = [
    ["measured", "Measured", evidence.measured ?? []],
    ["reconstructed", "Reconstructed", evidence.reconstructed ?? []],
  ];
  if (evidence.matchedReplay) {
    columns.push(["matched", "Matched replay", evidence.matched ?? []]);
  }
  if (state.liveMode) {
    columns.push(["illustrative", "Illustrative", [
      "Kernel sweep spans the step's wall-clock, not per-kernel CUPTI timing.",
    ]]);
  }
  columns.push(["unknown", "Unavailable", evidence.unavailable ?? []]);
  byId("evidence-grid").innerHTML = columns.map(([kind, title, values]) =>
    `<div class="evidence-column"><span class="badge ${kind}">${escapeMarkup(title)}</span><ul>${values.map((value) => `<li>${escapeMarkup(value)}</li>`).join("")}</ul></div>`
  ).join("");
}

function render() {
  renderHeader();
  renderStep();
  renderEvidence();
}

function showGraphArea() {
  byId("graph-idle").classList.add("hidden");
  byId("graph-scroll").classList.remove("hidden");
  byId("capture-status").classList.remove("hidden");
}

async function loadObject(object) {
  // Loading a sealed TraceBundle file is an explicit switch back to offline
  // viewing: stop treating incoming WS patches as belonging to this trace.
  state.liveMode = false;
  state.liveStepIds = new Map();
  state.liveFocusHash = null;
  state.liveStepActive = false;
  byId("output-connector")?.classList.remove("active");
  state.trace = validateTrace(object);
  state.stepIndex = 0;
  state.kernelIndex = 0;
  byId("error-panel").classList.add("hidden");
  showGraphArea();
  render();
}

async function loadUrl(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Trace fetch failed: HTTP ${response.status}`);
  await loadObject(await response.json());
}

// ---------------------------------------------------------------------------
// Live mode: chat box -> vLLM's own streaming endpoint (for text) and a
// WebSocket tail of the semantic ring (for the engine-step / scheduler /
// packed-row / kernel-graph animation). See the plan's "Why dual-stream text
// correlation" note: the ring's accepted_output_token record carries a token
// ID, not decoded text, so text comes from vLLM's SSE stream and the ring
// only confirms "this position landed," matched by output_position.
// ---------------------------------------------------------------------------

const params = new URLSearchParams(location.search);
const WS_URL = params.get("ws") ?? `ws://${location.hostname}:8089`;
const VLLM_BASE = params.get("vllm") ?? `http://${location.hostname}:8000`;
const MODEL_NAME = params.get("model") ?? "Qwen/Qwen3-14B";
// The shadow container (run-live-demo-shadow.sh) runs the sanitizer instead
// of CUPTI -- see the plan's mutual-exclusivity finding. Every chat request
// is mirrored here purely to generate GPU telemetry for the kernel-activity
// card; its reply is never shown.
const SHADOW_VLLM_BASE = params.get("shadowVllm") ?? `http://${location.hostname}:8001`;

function phaseFromRaw(raw) {
  return Number(raw) === 1 ? "prefill" : "decode";
}

function emptyLiveTrace() {
  return {
    schema: "GPU_OBSERVER_TRACE_V2",
    generatedBy: "gpu-observer live UI (not the cold exporter)",
    run: {
      model: MODEL_NAME,
      executionMode: "live",
      captureStatus: "live capture, single session — not a sealed offline trace",
    },
    query: { tokens: [], request: { messages: [] }, rendered_prompt: null },
    outputManifest: null,
    focus: { externalRequestId: null, internalRequestId: null, requestHash: null },
    evidence: {
      measured: ["Engine-step scheduling and packed-row slices, every request."],
      reconstructed: ["Scheduler→packed-row reordering, compared client-side."],
      matchedReplay: false,
      matched: [],
      unavailable: ["CUPTI per-kernel timing, SASS replay, exact tokenizer strings — offline only."],
    },
    metrics: { engineSteps: 0, cuptiKernels: 0 },
    steps: [],
  };
}

// Starts (or restarts) a fresh live trace. Previously this only reset state
// the first time it was called in a page session -- guarded by
// `if (state.liveMode) return` -- so every chat message after the first one
// kept appending its steps onto the SAME growing trace instead of starting
// clean, and the graph's step selector kept accumulating every step from
// every prior message in the session. This has exactly one caller
// (sendChatMessage, once per new message), so it should reset every time.
function ensureLiveTrace() {
  state.liveMode = true;
  state.trace = emptyLiveTrace();
  state.liveStepIds = new Map();
  state.stepIndex = 0;
  state.kernelIndex = 0;
  state.liveStepActive = false;
  byId("output-connector")?.classList.remove("active");
  byId("error-panel").classList.add("hidden");
  showGraphArea();
}

function getOrCreateLiveStep(stepId) {
  if (state.liveStepIds.has(stepId)) {
    return state.trace.steps[state.liveStepIds.get(stepId)];
  }
  const step = {
    id: stepId,
    phase: "decode",
    beginNs: null,
    endNs: null,
    wallUs: 0,
    scheduledTokens: 0,
    prefillTokens: 0,
    decodeTokens: 0,
    queueDepth: 0,
    activeRequests: 0,
    kvCacheUsagePermyriad: 0,
    schedulerOrderMismatches: 0,
    schedulerSlices: [],
    packedSlices: [],
    focusedPackedTokens: [],
    acceptedOutputTokens: [],
    kernels: [],
  };
  state.liveStepIds.set(stepId, state.trace.steps.length);
  state.trace.steps.push(step);
  return step;
}

function computeSchedulerOrderMismatches(step) {
  const length = Math.min(step.schedulerSlices.length, step.packedSlices.length);
  let mismatches = 0;
  for (let index = 0; index < length; index += 1) {
    if (step.schedulerSlices[index].requestId !== step.packedSlices[index].requestId) mismatches += 1;
  }
  step.schedulerOrderMismatches = mismatches;
}

// promptTokenCount comes from this turn's first request_slice patch's
// `tokens` field -- scheduler_output.num_scheduled_tokens for a freshly
// admitted request, straight off the Python scheduler with no extra
// instrumentation (see gpu_observer_semantic.py's _begin()). For the common
// case (prompt fits in one prefill step) that IS the real tokenized prompt
// length; with chunked prefill splitting a long prompt across steps this
// would only capture the first chunk, not re-verified live here.
function resolveLiveFocus(requestHash, promptTokenCount) {
  if (!state.pendingFocusReset) return;
  state.pendingFocusReset = false;
  state.liveFocusHash = requestHash;
  state.trace.focus.requestHash = requestHash;
  state.liveFocusPromptTokens = promptTokenCount;
}

function refreshLiveHeader() {
  state.trace.metrics.engineSteps = state.trace.steps.length;
  renderHeader();
}

function applyPatch(patch) {
  if (!state.liveMode) return; // a patch arrived while viewing a sealed trace; ignore it
  switch (patch.kind) {
    case "step_begin": {
      const step = getOrCreateLiveStep(patch.step);
      step.beginNs = patch.ts;
      step.scheduledTokens = patch.scheduled;
      step.prefillTokens = patch.prefill;
      step.decodeTokens = patch.decode;
      step.queueDepth = patch.queue;
      step.activeRequests = patch.active;
      step.kvCacheUsagePermyriad = patch.kv_permyriad;
      step.phase = patch.prefill > 0 && patch.decode > 0
        ? "mixed"
        : patch.prefill > 0 ? "prefill" : "decode";
      state.liveStepActive = true;
      state.stepIndex = state.trace.steps.length - 1;
      state.kernelIndex = 0;
      resetCuptiStepCounter();
      byId("output-connector")?.classList.add("active");
      byId("chat-waiting").textContent =
        `Step ${patch.step} scheduled — ${step.phase}, ${step.scheduledTokens} tokens, queue depth ${step.queueDepth}, ${step.activeRequests} active request(s).`;
      refreshLiveHeader();
      renderStep();
      break;
    }
    case "request_slice": {
      const step = getOrCreateLiveStep(patch.step);
      resolveLiveFocus(patch.request, patch.tokens);
      step.schedulerSlices.push({
        requestId: patch.request,
        phase: phaseFromRaw(patch.phase),
        scheduledTokens: patch.tokens,
      });
      renderStep();
      break;
    }
    case "packed_layout_begin":
      renderStep();
      break;
    case "packed_request_slice": {
      const step = getOrCreateLiveStep(patch.step);
      step.packedSlices.push({
        requestId: patch.request,
        packedIndex: patch.packed_index,
        rowBegin: patch.row_begin,
        rowEnd: patch.row_end,
        scheduledTokens: patch.tokens,
        phase: phaseFromRaw(patch.phase),
        authoritative: true,
      });
      computeSchedulerOrderMismatches(step);
      renderStep();
      break;
    }
    case "packed_token_row": {
      const step = getOrCreateLiveStep(patch.step);
      step.focusedPackedTokens.push({
        requestId: patch.request,
        packingGeneration: patch.generation,
        packedRow: patch.packed_row,
        sequencePosition: patch.sequence_position,
        tokenId: patch.token_id,
        phase: phaseFromRaw(patch.phase),
        timestampNs: patch.ts,
      });
      renderStep();
      break;
    }
    case "accepted_output_token": {
      const step = getOrCreateLiveStep(patch.step);
      step.acceptedOutputTokens.push({
        requestId: patch.request,
        outputPosition: patch.output_position,
        tokenId: patch.token_id,
        timestampNs: patch.ts,
      });
      onGpuConfirmedToken(patch.output_position);
      renderStep();
      break;
    }
    case "step_end": {
      const step = getOrCreateLiveStep(patch.step);
      step.endNs = patch.ts;
      if (step.beginNs != null) {
        step.wallUs = Number((BigInt(step.endNs) - BigInt(step.beginNs)) / 1000n);
      }
      state.liveStepActive = false;
      byId("output-connector")?.classList.remove("active");
      byId("chat-waiting").textContent = `Step ${patch.step} complete in ${step.wallUs.toFixed(1)} µs. Waiting for next step…`;
      renderStep();
      break;
    }
    case "dropped_total":
      if (Number(patch.count) > 0) {
        byId("ws-status").title = `${patch.count} ring records dropped (buffer overload) since capture start.`;
      }
      break;
    default:
      break; // unrecognized patch kind: ignore rather than throw, matching the ring's own forward-compat stance
  }
}

function connectWebSocket() {
  const status = byId("ws-status");
  let socket;
  try {
    socket = new WebSocket(WS_URL);
  } catch (error) {
    status.textContent = "disconnected";
    status.className = "status disconnected";
    scheduleReconnect();
    return;
  }
  state.ws = socket;
  socket.addEventListener("open", () => {
    status.textContent = "connected";
    status.className = "status connected";
  });
  socket.addEventListener("message", (event) => {
    try {
      applyPatch(JSON.parse(event.data));
    } catch (error) {
      // A malformed patch must never take down the live view.
      console.warn("gpu-observer: dropped malformed WS patch", error);
    }
  });
  socket.addEventListener("close", () => {
    status.textContent = "disconnected";
    status.className = "status disconnected";
    scheduleReconnect();
  });
  socket.addEventListener("error", () => socket.close());
}

function scheduleReconnect() {
  if (state.wsReconnectTimer) return;
  state.wsReconnectTimer = setTimeout(() => {
    state.wsReconnectTimer = null;
    connectWebSocket();
  }, 2000);
}

setInterval(() => {
  if (!state.liveStepActive) return;
  // Real CUPTI kernel_launch events arrive continuously via cupti-activity.js
  // (often many per millisecond during decode); re-rendering the SVG graph
  // on every single message would be wasteful, so this throttles the actual
  // DOM update to a fixed cadence while cupti-activity.js's own internal
  // state updates immediately on every message.
  renderGraph();
}, GPU_REFRESH_INTERVAL_MS);

// ---- Chat: text from vLLM's own streaming endpoint, correlated with ring
// accepted_output_token events by output_position (see plan). ----

let chatConfirmedCount = 0;
let chatTotalCount = 0;

function onGpuConfirmedToken(outputPosition) {
  if (Number(outputPosition) < chatTotalCount) chatConfirmedCount += 1;
  updateChatConfirmedBadge();
}

function updateChatConfirmedBadge() {
  byId("chat-confirmed").textContent =
    chatTotalCount === 0 ? "" : `${chatConfirmedCount}/${chatTotalCount} tokens GPU-confirmed`;
}

// Mirrors the same prompt to the shadow (sanitizer) container so its GPU
// telemetry reflects the same request the visible chat is asking about.
// The two containers are independently scheduled processes -- this is a
// best-effort approximate time alignment, not a causal join (confirmed not
// achievable even offline -- see the plan). Errors here must never surface
// in the visible chat: the shadow request is a bonus telemetry source, not
// a required part of the primary flow.
async function sendShadowRequest(text, signal) {
  try {
    const response = await fetch(`${SHADOW_VLLM_BASE}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify({
        model: MODEL_NAME,
        messages: [{ role: "user", content: text }],
        stream: true,
      }),
    });
    if (!response.ok || !response.body) return;
    const reader = response.body.getReader();
    for (;;) {
      const { done } = await reader.read(); // drain and discard -- only the shadow container's GPU activity matters
      if (done) break;
    }
  } catch (error) {
    // Shadow container unreachable/not running: the kernel-activity card
    // will simply show "disconnected", same as any other WS drop. Silent
    // here on purpose -- see comment above.
  }
}

async function sendChatMessage(text) {
  if (state.chatBusy || !text.trim()) return;
  state.chatBusy = true;
  byId("chat-send").disabled = true;
  byId("chat-stop").disabled = false;
  // Aborting this controller closes both fetches; vLLM detects the client
  // disconnect on a streaming request and cancels the in-flight generation
  // server-side too -- this actually stops the decode loop, not just the
  // UI, which is the point (fast iteration on UI changes without waiting
  // out a full generation every time).
  const controller = new AbortController();
  state.activeAbort = controller;
  ensureLiveTrace();
  resetCuptiQueryCounters();
  state.pendingFocusReset = true;
  state.liveFocusHash = null;
  state.liveFocusPromptTokens = null;
  state.trace.focus.requestHash = null;
  state.trace.query.request.messages = [{ role: "user", content: text }];
  chatConfirmedCount = 0;
  chatTotalCount = 0;
  byId("chat-confirmed").textContent = "";
  const reply = byId("chat-reply");
  reply.textContent = "";
  byId("chat-waiting").textContent = "Request sent — waiting for the scheduler to admit it…";
  render();

  sendShadowRequest(text, controller.signal); // fire-and-forget: generates GPU telemetry on the shadow container, never shown

  try {
    const response = await fetch(`${VLLM_BASE}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        model: MODEL_NAME,
        messages: [{ role: "user", content: text }],
        stream: true,
      }),
    });
    if (!response.ok || !response.body) {
      throw new Error(`vLLM request failed: HTTP ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() ?? "";
      for (const rawEvent of events) {
        const line = rawEvent.trim();
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") continue;
        let chunk;
        try {
          chunk = JSON.parse(payload);
        } catch (error) {
          continue; // partial/malformed SSE frame; skip rather than corrupt the reply
        }
        const delta = chunk.choices?.[0]?.delta?.content;
        if (delta) {
          reply.textContent += delta;
          chatTotalCount += 1;
          updateChatConfirmedBadge();
          // Real content boundary (Qwen3's own <think>/</think> tags) driving
          // the kernel-count graphic's thinking/response split -- see
          // cupti-activity.js's inThinkingPhase for why this is a
          // reconstructed correlation, not a measured one.
          const hasOpenThink = reply.textContent.includes("<think>");
          const hasCloseThink = reply.textContent.includes("</think>");
          setThinkingPhase(hasOpenThink && !hasCloseThink);
        }
      }
    }
    byId("chat-waiting").textContent = state.liveStepActive
      ? byId("chat-waiting").textContent
      : "Response complete.";
  } catch (error) {
    byId("chat-waiting").textContent = error.name === "AbortError"
      ? "Stopped — generation cancelled server-side."
      : `Live request failed: ${error.message}`;
  } finally {
    state.chatBusy = false;
    state.activeAbort = null;
    byId("chat-send").disabled = false;
    byId("chat-stop").disabled = true;
  }
}

byId("chat-stop").addEventListener("click", () => {
  state.activeAbort?.abort();
});

// Escape as the keyboard shortcut, not Ctrl+C: browsers reserve Ctrl+C for
// copy and won't let a page reliably intercept it without breaking that
// convention (and stealing it while text is selected would be actively
// hostile). Escape is the standard web pattern for "cancel the in-flight
// thing" and works globally, not just while the chat input is focused --
// useful for fast UI-iteration loops where you want to kill a generation
// without clicking back into the form first.
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && state.activeAbort) {
    state.activeAbort.abort();
  }
});

byId("chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = byId("chat-input");
  const text = input.value;
  input.value = "";
  sendChatMessage(text);
});

// Hosted/offline mode (?offline=1). The page is static -- ui/ is just files,
// and renderOfflineGraph needs no backend at all -- so it can be served
// publicly (GitHub Pages) with a sealed TraceBundle via ?trace=. But three
// panels only mean anything against a live container: the chat box, the
// shadow-container SM-occupancy card, and the decoded-output card fed by the
// chat stream. Left alone they actively mislead: the WS status pill sits on
// "connecting" forever against nothing, and the SM grid renders 48 dead
// cells, which reads as broken rather than as not-applicable. So skip every
// live connection and hide exactly those panels, leaving a coherent
// sealed-trace viewer. Everything that makes the offline view worth showing
// -- the lane graph, kernel ownership, SASS microscope, evidence tiers --
// is untouched, because none of it depends on a socket.
const offlineMode = params.get("offline") === "1";
if (offlineMode) {
  for (const selector of [".chat-card", ".kernel-activity-card", ".output-card", "#output-connector"]) {
    document.querySelector(selector)?.classList.add("hidden");
  }
} else {
  connectWebSocket();
  connectKernelActivity();
  connectCuptiActivity();
}

byId("trace-file").addEventListener("change", async (event) => {
  try {
    const file = event.target.files?.[0];
    if (!file) return;
    if (file.size <= 0 || file.size > 64 * 1024 * 1024) throw new Error("Trace file must be between 1 byte and 64 MiB.");
    await loadObject(JSON.parse(await file.text()));
  } catch (error) {
    byId("error-panel").textContent = error.message;
    byId("error-panel").classList.remove("hidden");
  }
});

// No trace loads automatically: the graph starts empty (#graph-idle) rather
// than snapping in a default/fixture graph the user never asked for. An
// explicit ?trace= still works for pointing at a specific sealed bundle;
// otherwise the graph only appears once the user opens a file or sends a
// live chat message (ensureLiveTrace / loadObject both call showGraphArea).
// A real sealed capture, so anyone opening the hosted viewer sees measured
// data without needing a bundle of their own. Same path as ?trace=.
byId("load-sample").addEventListener("click", () => {
  byId("load-sample").disabled = true;
  loadUrl("./data/sample-qwen3-14b-trace-v2.json")
    .catch((error) => {
      byId("error-panel").textContent = `${error.message} Serve ui/ over HTTP, or choose a local TraceBundle v2 file.`;
      byId("error-panel").classList.remove("hidden");
    })
    .finally(() => { byId("load-sample").disabled = false; });
});

const explicitTraceUrl = new URLSearchParams(location.search).get("trace");
if (explicitTraceUrl) {
  loadUrl(explicitTraceUrl).catch((error) => {
    byId("error-panel").textContent = `${error.message} Serve ui/ through HTTP or choose a local TraceBundle v2 file.`;
    byId("error-panel").classList.remove("hidden");
  });
}

