import { renderCausalGraph } from "./causal-graph.js?v=v2g5";
import { connectKernelActivity } from "./kernel-activity.js?v=v2g5";
import { installReplay, showReplayMarker } from "./replay.js?v=v2g5";
import { connectCuptiActivity, getCuptiSnapshot, resetCuptiStepCounter, resetCuptiQueryCounters, resetCuptiAll, setThinkingPhase, setCuptiFrameTap, feedCuptiEvent } from "./cupti-activity.js?v=v2g5";
import { createSessionRecorder } from "./session-recorder.js?v=v2g5";
const GPU_REFRESH_INTERVAL_MS = 150; // re-render cadence for freshly arrived real CUPTI data, not a paced sweep

const state = {
  trace: null,
  stepIndex: 0,
  kernelIndex: 0,
  // Live-mode fields. Unused (stay at defaults) when viewing a sealed
  // TraceBundle file -- offline viewing behaves exactly as before.
  liveMode: false,
  // liveMode only turns true once a chat message starts a live trace. livePage
  // is true for the whole session of a non-offline page, so the engine's state
  // can be read and shown before anything has been sent.
  livePage: false,
  liveStepIds: new Map(), // step-id string -> index into trace.steps, for O(1) patch application
  liveFocusHash: null, // which request's hash this chat turn is about; see resolveLiveFocus()
  liveFocusPromptTokens: null, // real prompt token count, captured off this turn's first request_slice patch -- see resolveLiveFocus()
  pendingFocusReset: false,
  liveStepActive: false,
  // Transport. `pinned` keeps the playhead on the newest live step; any manual
  // seek detaches it (the log-viewer contract) and "jump to live" re-attaches.
  transport: { playing: false, speed: 1, pinned: true, timer: null },
  // true/false once the control endpoint answers, null when it is unreachable
  // (then pause is view-only and the engine keeps generating).
  enginePaused: false,
  // The transport over a recording: either the session this page just
  // recorded itself (the normal path -- your own query, rewindable) or a
  // capture loaded from a file via ?replay= on a machine with no GPU.
  replay: null,
  ws: null,
  wsReconnectTimer: null,
  chatBusy: false,
  activeAbort: null, // AbortController for the in-flight chat request, if any -- see the Stop button
};
const byId = (id) => document.getElementById(id);
// This folder IS the v2 shell: ui/ keeps the shipped page untouched, and the
// redesign lands here (see the plan). No feature flag needed.
const V2 = true;
const SPINE_TICK = 4;        // viewBox units per step: fixed, so live appends one rect instead of rebuilding all
const SPINE_TOKEN_REF = 128; // tokens that reach full tick height on a log scale
const PLAY_BASE_MS = 250;    // one step per 250ms at 1x
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
  positionInputConnector();
}

// The live graph has no Query node of its own -- the chat box above IS the
// query -- so this arrow joins the two. It used to be pinned under the graph's
// tokens node (renderLiveGraph tags it data-entry="prompt-tokens"), but that
// node sits at the far left of the lane layout, which left the arrow stranded
// off to one side under a centred chat box; it now sits centred, matching the
// chat box it descends from. The tokens node still decides *whether* it shows,
// so the arrow only appears once there is something for the prompt to land on.
// Shown only while that live graph is on screen: a sealed trace's query never
// came from this chat box, and ?offline=1 hides the chat box entirely. It flows while a request is in flight (Stop is
// enabled) and the tokenized prompt length has not arrived yet; keying off
// Stop means an error or abort settles it too, via sendChatMessage's finally.
function positionInputConnector() {
  const connector = byId("input-connector");
  if (!connector) return;
  const entry = state.liveMode
    ? byId("causal-graph").querySelector('[data-entry="prompt-tokens"]')
    : null;
  if (!entry) {
    connector.classList.add("hidden");
    connector.classList.remove("active");
    return;
  }
  connector.classList.remove("hidden");
  connector.style.removeProperty("--entry-x"); // fall back to the CSS 50% centre
  const inFlight = !byId("chat-stop").disabled;
  connector.classList.toggle("active", inFlight && state.liveFocusPromptTokens == null);
}

function selectStep(index, { manual = false } = {}) {
  state.stepIndex = Math.max(0, Math.min(index, state.trace.steps.length - 1));
  state.kernelIndex = 0;
  if (manual && V2) state.transport.pinned = false;
  renderStep();
}

// ---- v2 transport ---------------------------------------------------------

// "Moving" is what the eye sees: stepping through buffered steps, or riding the
// newest live step. Pause has to stop BOTH, otherwise a live view keeps being
// yanked forward by step_begin and the DAG keeps repainting, which is what
// "pause does nothing" looked like.
function isMoving() {
  return state.transport.playing || (state.liveMode && state.transport.pinned);
}

// Frozen means the stage holds still. Capture keeps running: steps keep being
// appended and new ticks keep appearing on the spine, so you can see data still
// arriving while the view stays put.
function isFrozen() {
  // A replay drives the graph itself -- seeking and playing both have to
  // repaint, whatever the live playhead was doing when the recording ended.
  if (state.replay) return false;
  return state.liveMode && !state.transport.pinned && !state.transport.playing;
}

function updatePlayButton() {
  const button = byId("tp-play");
  if (!button) return;
  const moving = state.replay ? state.replay.playing : isMoving();
  button.textContent = moving ? "\u275A\u275A" : "\u25B6";
  button.setAttribute("aria-pressed", moving ? "true" : "false");
  button.setAttribute("aria-label", moving ? "Pause" : "Play");
}

function stopPlayback() {
  if (state.transport.timer) clearInterval(state.transport.timer);
  state.transport.timer = null;
  state.transport.playing = false;
  updatePlayButton();
}

function startPlayback() {
  if (!state.trace?.steps.length) return;
  stopPlayback();
  state.transport.playing = true;
  updatePlayButton();
  state.transport.timer = setInterval(() => {
    const last = state.trace.steps.length - 1;
    if (state.stepIndex >= last) {
      // Live: playback caught up with capture, so hand control back to the
      // live playhead rather than stopping dead at the newest step.
      if (state.liveMode) { state.transport.pinned = true; stopPlayback(); renderSpine(); return; }
      stopPlayback();
      return;
    }
    selectStep(state.stepIndex + 1, { manual: true });
  }, Math.max(40, PLAY_BASE_MS / state.transport.speed));
}

// Pausing the view is only half the story: the engine keeps generating unless
// vLLM's own scheduler is paused. EngineCore exposes no RPC for that, so the
// overlay watches a flag file in the bind-mounted run directory and this talks
// to the small host endpoint that writes it (ui-v2/tools/pause-control.py).
// PAUSED_ALL sets the scheduler's token budget to zero: no request is
// scheduled, no kernel launches, KV cache is kept, and resume continues the
// same token stream.
// The pause flag lives in a file on the host and outlives the page: pause the
// engine, close the tab, and it is still paused on the next load. state
// .enginePaused starts false, so without this the UI would assert "running"
// about an engine it never asked, and every message sent would hang with no
// explanation -- which is exactly what a left-over pause looks like from the
// chat box. Ask once at startup and show what is actually true.
async function syncEnginePaused() {
  if (!state.livePage) return;
  try {
    const response = await fetch(`${CONTROL_BASE}/state`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.enginePaused = Boolean((await response.json()).paused);
  } catch (error) {
    state.enginePaused = null; // no control endpoint: the freeze is view-only
  }
  renderSpine();
}

async function setEnginePaused(paused) {
  if (!state.liveMode) return; // a sealed trace has no engine to pause
  try {
    const response = await fetch(`${CONTROL_BASE}/${paused ? "pause" : "resume"}`, { method: "POST" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.enginePaused = Boolean((await response.json()).paused);
  } catch (error) {
    state.enginePaused = null; // no control endpoint: the freeze is view-only
  }
  renderSpine();
}

// applyPatch() drops every frame unless state.liveMode is set, and only
// ensureLiveTrace() sets it -- which until now only sendChatMessage() called.
// So pressing play fed frames into a view that was throwing them away, and the
// recording appeared to do nothing. A replay has to open the same empty live
// trace a real request would before its first frame arrives.
function startReplay() {
  if (!state.replay) return;
  // A session recording is armed at its end, so the first press of play has
  // nothing left to deliver. Rewind to the start, then run it.
  if (state.replay.atEnd?.()) state.replay.seekToToken(0);
  if (state.replay.cursor === 0) {
    ensureLiveTrace();
    resetCuptiQueryCounters();
    state.trace.query.request.messages = [
      { role: "user", content: state.replay.manifest?.prompt ?? "(recorded prompt)" },
    ];
    render();
  }
  state.replay.start();
}

function togglePlayback() {
  // With a recording loaded, play/pause means the recording -- there is no
  // engine to pause and no live playhead to detach from.
  if (state.replay) {
    if (state.replay.playing) state.replay.pause();
    else startReplay();
    updatePlayButton();
    return;
  }
  if (isMoving()) {
    // Freeze: detach from the live playhead as well as stopping playback.
    stopPlayback();
    state.transport.pinned = false;
    setEnginePaused(true);
    renderSpine();
    return;
  }
  setEnginePaused(false);
  startPlayback();
}

// Two different speeds share one button. Scrubbing recorded steps wants to go
// FASTER than real time; replaying a capture wants to go much slower, because
// the thing worth seeing -- 440 kernel launches inside one 106ms step -- is
// otherwise over before the browser can paint it twice.
const STEP_SPEEDS = [1, 2, 4];
const REPLAY_SPEEDS = [1, 0.5, 0.1, 0.02];

function speedLabel(value) {
  return value >= 1 ? `${value}x` : `${value}`.replace(/^0/, "") + "x";
}

function cycleSpeed(direction = 1) {
  const replay = state.replay;
  const speeds = replay ? REPLAY_SPEEDS : STEP_SPEEDS;
  const current = replay ? replay.speed : state.transport.speed;
  const index = speeds.indexOf(current);
  const next = speeds[((index < 0 ? 0 : index) + direction + speeds.length) % speeds.length];
  const button = byId("tp-speed");
  if (button) button.textContent = speedLabel(next);
  if (replay) {
    replay.setSpeed(next);
    const readout = byId("replay-speed");
    if (readout) readout.textContent = speedLabel(next);
    return;
  }
  state.transport.speed = next;
  if (state.transport.playing) startPlayback(); // restart at the new cadence
}

function spineIndexFromEvent(event) {
  const svg = byId("spine");
  const rect = svg.getBoundingClientRect();
  if (!rect.width || !state.trace?.steps.length) return 0;
  const ratio = (event.clientX - rect.left) / rect.width;
  return Math.round(ratio * (state.trace.steps.length - 1));
}

// Ticks are appended, never rebuilt: each step owns a fixed SPINE_TICK slice of
// the viewBox and the viewBox grows, so a 400-step live run costs one <rect>
// per step instead of re-laying-out every tick at the 150ms render cadence.
function updateTransportEnabled() {
  const count = state.trace?.steps.length ?? 0;
  for (const id of ["tp-prev", "tp-play", "tp-next", "tp-speed"]) {
    const button = byId(id);
    if (button) button.disabled = count === 0;
  }
  // In a replay the speed button sets the playback rate, not the step cadence,
  // so it has to work before the first frame has been delivered -- picking the
  // speed is the first thing anyone does with a recording.
  for (const id of ["tp-speed", "tp-play", "tp-prev", "tp-next"]) {
    const button = byId(id);
    if (button && state.replay) button.disabled = false;
  }
  // With no steps the full-height spine is 64px of empty box. Collapse it to a
  // baseline so the shell reads as an axis awaiting data, not a dead band.
  byId("timeline")?.classList.toggle("empty", count === 0);
}

function renderSpine() {
  const svg = byId("spine");
  if (!svg) return;
  const steps = state.trace?.steps ?? [];
  updateTransportEnabled();
  const built = Number(svg.dataset.built ?? 0);
  if (steps.length < built) { svg.innerHTML = ""; svg.dataset.built = "0"; }
  const from = Number(svg.dataset.built ?? 0);
  if (steps.length > from) {
    const parts = [];
    for (let i = from; i < steps.length; i += 1) {
      const step = steps[i];
      const value = step.scheduledTokens || step.kernels.length || 1;
      const height = 14 + 76 * Math.min(1, Math.log1p(value) / Math.log1p(SPINE_TOKEN_REF));
      parts.push(`<rect class="tick ${escapeMarkup(step.phase || "decode")}" data-i="${i}" x="${i * SPINE_TICK}" y="${(100 - height).toFixed(1)}" width="${SPINE_TICK - 1}" height="${height.toFixed(1)}"/>`);
    }
    svg.insertAdjacentHTML("beforeend", parts.join(""));
    svg.dataset.built = String(steps.length);
    const head = svg.querySelector(".head") ?? (() => {
      svg.insertAdjacentHTML("beforeend", '<rect class="head" y="0" width="1.4" height="100"/>');
      return svg.querySelector(".head");
    })();
    head.parentNode.appendChild(head); // keep the playhead on top of newly appended ticks
  }
  svg.setAttribute("viewBox", `0 0 ${Math.max(1, steps.length) * SPINE_TICK} 100`);
  const head = svg.querySelector(".head");
  if (head) head.setAttribute("x", String(state.stepIndex * SPINE_TICK));
  const previous = svg.querySelector("rect.tick.current");
  if (previous) previous.classList.remove("current");
  svg.querySelector(`rect.tick[data-i="${state.stepIndex}"]`)?.classList.add("current");
  const engineNote = state.livePage && state.enginePaused === true ? " · engine paused (PAUSED_ALL)"
    : state.livePage && state.enginePaused === null ? " · view only, no engine control"
    : "";
  const step = steps[state.stepIndex];
  byId("spine-label").textContent = (step
    ? `step ${step.id} · ${step.phase} · ${step.scheduledTokens} tokens · ${step.kernels.length} kernels · ${state.stepIndex + 1}/${steps.length}`
    : "no steps yet") + engineNote;
  byId("tp-live").classList.toggle("hidden", !(state.liveMode && !state.transport.pinned));
  updatePlayButton();
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
  if (state.replay) {
    // renderHeader runs on every patch, so a one-off line set when the replay
    // loaded was overwritten by the first frame and the page went back to
    // calling itself live.
    const prompt = state.replay.manifest?.prompt;
    const where = state.replay.live ? "Replay of this session" : "Replay of a capture";
    byId("run-subtitle").textContent =
      `${where} · ${run.model ?? "unknown model"} · ${state.trace.steps.length} engine steps`
      + (prompt ? ` · "${prompt}"` : "");
    return;
  }
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
  if (preferred >= 0 && !state.transport.playing && !state.trace.steps[state.stepIndex]?.kernels.length) state.stepIndex = preferred;

  if (V2) {
    byId("timeline").classList.remove("hidden");
    byId("graph-step-select").closest(".step-select-label")?.classList.add("hidden");
    renderSpine();
    return;
  }

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
    // The kernel-ownership card was removed from this shell; phase 2 brings it
    // back inside the inspector drawer, so these renders are now optional.
    const idleDetail = byId("kernel-detail");
    if (idleDetail) idleDetail.innerHTML = '<div class="empty-state">No engine step yet.</div>';
    byId("ownership-list")?.replaceChildren();
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
  if (!detail || !owners) return; // card not present in this shell
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
  if (!status || !content) return; // SASS card is not part of this shell
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
  const grid = byId("evidence-grid");
  if (!grid) return; // evidence tiers are not part of this shell
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
  grid.innerHTML = columns.map(([kind, title, values]) =>
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
const CONTROL_BASE = params.get("control") ?? `http://${location.hostname}:8091`;
// ?replay=<dir or session.ndjson> swaps the live transport for a recording.
const replayUrl = params.get("replay");
// The shadow container is opt-in (?shadow=1). Its card shows real block-entry
// events, but live it cannot place an event in the step it ran in: delivery
// is batched and the events carry only a device-clock timestamp, so its
// per-step counters and request attribution are joined by arrival time.
// It also costs a second 14B container. The default demo therefore runs the
// primary alone; ?shadow=1 restores the card, its two sockets and the mirror.
const shadowEnabled = params.get("shadow") === "1";

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
      if (state.transport.pinned) {
        state.stepIndex = state.trace.steps.length - 1;
        state.kernelIndex = 0;
      }
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
      const patch = JSON.parse(event.data);
      recorder.capture("semantic", patch);
      // The semantic ring broadcasts every engine step on the box, not just
      // this page's request. While a recording is being inspected it owns the
      // view, so unrelated live frames would scribble over the step being
      // examined. Asking a new question clears the recording and live resumes.
      if (!state.replay?.live) applyPatch(patch);
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
  if (isFrozen()) return; // paused: hold the stage still while capture continues
  // Real CUPTI kernel_launch events arrive continuously via cupti-activity.js
  // (often many per millisecond during decode); re-rendering the SVG graph
  // on every single message would be wasteful, so this throttles the actual
  // DOM update to a fixed cadence while cupti-activity.js's own internal
  // state updates immediately on every message.
  renderGraph();
}, GPU_REFRESH_INTERVAL_MS);

// Every live frame is kept while a query is in flight, so the answer you just
// watched can be rewound token by token with the kernels that produced it. See
// session-recorder.js for why this is the only honest way to show a sweep that
// is over before the browser can paint it.
const recorder = createSessionRecorder();
setCuptiFrameTap((event) => recorder.capture("cupti", event));

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

// A question typed while a file capture was loaded has to survive the reload
// that leaves replay mode, because ?replay= is a page mode, not a panel.
const PENDING_PROMPT_KEY = "gpu-observer:pending-prompt";

async function sendChatMessage(text) {
  // ?replay= has no engine behind it, so there is nothing here to ask. The
  // submit handler sends these to live instead; this stays as a guard because
  // running the live path against the fetch shim appended every delta twice,
  // once from the shim's reader and once from the player's own callback.
  if (replayUrl) return;
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
  // A new question invalidates the previous recording: the transport must not
  // rewind into an answer that is no longer on screen.
  state.replay = null;
  document.getElementById("replay-marker")?.remove();
  recorder.begin(text);
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

  // A paused engine accepts the request and then sits on it, so the page showed
  // "waiting for the scheduler to admit it" indefinitely -- which reads as a
  // hung model rather than as a pause somebody left on. Asking a question means
  // you want it answered, so clear the pause first. The control endpoint is
  // re-read rather than trusting state.enginePaused, which goes stale the moment
  // anything else touches the flag (another tab, an earlier session, a probe).
  await syncEnginePaused();
  if (state.enginePaused) {
    byId("chat-waiting").textContent = "Engine was paused — resuming before the request can be scheduled…";
    await setEnginePaused(false);
  }

  if (shadowEnabled) sendShadowRequest(text, controller.signal); // fire-and-forget: generates GPU telemetry on the shadow container, never shown

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
          recorder.capture("chat", { kind: "delta", delta: chunk });
          chatTotalCount += 1;
          updateChatConfirmedBadge();
          // Held back in a recording: the frame is kept, but appending it now
          // would fight the rewind -- the text grew while the user pressed back.
          if (state.replay?.live) continue;
          reply.textContent += delta;
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
    // The edge-flow animation is driven by "a step is in flight", which
    // step_end clears. That step_end can go missing at exactly this moment: on
    // an abort it may never be emitted, and once a recording is armed the
    // socket gate stops applying live frames, so a step_end arriving after the
    // last step_begin is dropped. Either way the flag stayed set with nothing
    // generating, and the arrows kept flowing until the next request. Whether
    // the flag is still set here is a race against the final frame, which is
    // why this was intermittent rather than always wrong. The generation is
    // over, so say so outright instead of waiting for a frame to say it.
    state.liveStepActive = false;
    byId("output-connector")?.classList.remove("active");
    // Stopped generations are worth rewinding too -- the frames up to the stop
    // are real -- so this runs whether the stream finished or was aborted.
    recorder.stop();
    armReplayOfThisSession();
    renderGraph(); // one last paint, with nothing in flight, to clear the flow
  }
}

// Hands the transport a player over what this page just recorded. The handlers
// are the same consumers the live sockets fed, so rewinding re-drives the real
// view code: applyPatch rebuilds the steps and the layer sweep, the CUPTI
// classifier re-sees every launch in order, and the text comes back one piece
// at a time.
function buildSessionPlayer() {
  const player = recorder.buildPlayer({
    semantic: (msg) => applyPatch(msg),
    cupti: (msg) => feedCuptiEvent(msg),
    chat: (piece) => {
      const reply = byId("chat-reply");
      reply.textContent += piece;
      const open = reply.textContent.includes("<think>");
      const close = reply.textContent.includes("</think>");
      setThinkingPhase(open && !close);
    },
  });
  // A seek cannot un-apply frames -- every consumer accumulates -- so it
  // replays from the start, and everything that accumulates clears first.
  player.onReset = () => {
    ensureLiveTrace();
    resetCuptiAll();
    state.trace.query.request.messages = [{ role: "user", content: player.manifest.prompt }];
    byId("chat-reply").textContent = "";
    setThinkingPhase(false);
  };
  player.onProgress = () => { renderSpine(); updatePlayButton(); };
  return player;
}

// Arms the transport over what has been recorded SO FAR. Rewind used to become
// available only once the generation finished, and a generation is long -- 37s
// measured for "Name three colors.", 284 engine steps -- so for the whole time
// anyone would actually want to step back through it, pressing back moved the
// playhead over captured steps while live frames kept appending text, and the
// answer grew instead of shrinking. There is no reason to wait: the frames are
// already in hand, so the recording is built on demand the moment it is asked
// for. Recording continues in the background, so returning to live loses
// nothing that arrived while you were looking backwards.
function armReplayOfThisSession() {
  if (state.replay?.live) return true; // already armed
  if (recorder.count() === 0) return false; // nothing arrived yet
  const player = buildSessionPlayer();
  state.replay = player;
  state.transport.pinned = false;
  stopPlayback();
  // What is on screen is the latest frame, so the recording starts at its end:
  // the first rewind takes one token off what you are looking at.
  player.assumePlayed();
  showReplayMarker(player);
  updateTransportEnabled();
  updatePlayButton();
  renderHeader();
  renderSpine();
  return true;
}

// Rebuilds over everything recorded, including the frames that arrived while
// the view was held back, runs it to the end, and hands the view back to the
// live stream. Without the rebuild, returning to live would resume from a
// recording that stopped where it was armed.
function returnToLiveFromRecording() {
  if (!state.replay?.live) return;
  const player = buildSessionPlayer();
  player.seekToToken(player.tokenCount);
  state.replay = null;
  document.getElementById("replay-marker")?.remove();
  state.transport.pinned = true;
  updateTransportEnabled();
  updatePlayButton();
  renderHeader();
  renderSpine();
  renderStep();
}

// Transport wiring. Seeks are flagged manual so they unpin the live playhead;
// the spine itself is the only pointer surface, and keys mirror a media player.
function installTransport() {
  const svg = byId("spine");
  if (!svg) return;
  // Show the instrument shell immediately. Waiting for the first trace made a
  // fresh live load look exactly like the old page, with no transport at all.
  byId("timeline").classList.remove("hidden");
  updateTransportEnabled();
  const seek = (event) => {
    if (!state.trace?.steps.length) return;
    selectStep(spineIndexFromEvent(event), { manual: true });
  };
  let dragging = false;
  svg.addEventListener("pointerdown", (event) => { dragging = true; svg.setPointerCapture?.(event.pointerId); seek(event); });
  svg.addEventListener("pointermove", (event) => { if (dragging) seek(event); });
  svg.addEventListener("pointerup", (event) => { dragging = false; svg.releasePointerCapture?.(event.pointerId); });
  svg.addEventListener("pointercancel", () => { dragging = false; });
  svg.addEventListener("click", seek); // synthetic clicks (tests, assistive tech)

  // In a replay these step the RECORDING, one accepted output token at a time:
  // back one is a backspace on the decoded text, and because the seek re-feeds
  // the capture to that point the kernels, layer sweep and stage counts go back
  // with it, in the order they really happened. Outside a replay they just move
  // the playhead over already-captured steps.
  const step = (delta) => {
    // Stepping back means "take a token off what I am looking at", whether or
    // not the generation has finished. If a recording is not armed yet, arming
    // it is what makes that possible.
    if (!state.replay && delta < 0) armReplayOfThisSession();
    if (state.replay) {
      state.replay.seekToToken(state.replay.tokensEmitted() + delta);
      renderSpine();
      updatePlayButton();
      return;
    }
    if (state.trace?.steps.length) selectStep(state.stepIndex + delta, { manual: true });
  };
  byId("tp-play").addEventListener("click", togglePlayback);
  byId("tp-prev").addEventListener("click", () => step(-1));
  byId("tp-next").addEventListener("click", () => step(1));
  byId("tp-speed").addEventListener("click", () => cycleSpeed(1));
  byId("tp-live").addEventListener("click", () => {
    setEnginePaused(false); // following live again implies the engine should run
    if (state.replay?.live) {
      returnToLiveFromRecording();
      return;
    }
    state.transport.pinned = true;
    if (state.trace?.steps.length) selectStep(state.trace.steps.length - 1);
  });

  window.addEventListener("keydown", (event) => {
    if (event.target instanceof Element && event.target.closest("input, textarea, select")) return;
    const jump = event.shiftKey ? 10 : 1;
    if (event.key === " ") { event.preventDefault(); togglePlayback(); }
    else if (event.key === "ArrowLeft") { event.preventDefault(); step(-jump); }
    else if (event.key === "ArrowRight") { event.preventDefault(); step(jump); }
    else if (event.key === "[") cycleSpeed(-1);
    else if (event.key === "]") cycleSpeed(1);
  });
}
installTransport();

byId("chat-stop").addEventListener("click", () => {
  state.activeAbort?.abort();
});

window.addEventListener("resize", positionInputConnector);
byId("graph-scroll").addEventListener("scroll", positionInputConnector);
new MutationObserver(positionInputConnector)
  .observe(byId("chat-stop"), { attributes: true, attributeFilter: ["disabled"] });

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

// The live page records itself (armReplayOfThisSession), so replay needs no
// entry point of its own: you ask your question and then rewind it. The button
// that used to be here loaded a committed capture of a different prompt, which
// put someone else's question in the input box. ?replay= still loads a capture
// for hosting the demo with no GPU attached; it is not a click away any more.
// Ships hidden, so there is no flash of a button on the normal live page.
const replayButton = byId("open-replay");
if (replayButton && replayUrl) {
  replayButton.classList.remove("hidden");
  replayButton.addEventListener("click", () => {
    const params = new URLSearchParams(location.search);
    params.delete("replay");
    location.search = params.toString();
  });
}

byId("chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = byId("chat-input");
  const text = input.value;
  if (!text.trim()) return;
  input.value = "";
  // Asking a question while a capture is loaded means you want it answered, so
  // leave the recording rather than refusing. ?replay= is in the URL, so this
  // is a navigation; the prompt rides along and is sent on the way back in.
  if (replayUrl) {
    try {
      sessionStorage.setItem(PENDING_PROMPT_KEY, text);
    } catch (error) {
      // Private mode / blocked storage: still leave replay, just unprompted.
    }
    const params = new URLSearchParams(location.search);
    params.delete("replay");
    location.search = params.toString();
    return;
  }
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
  for (const selector of [".chat-card", ".kernel-activity-card", ".output-card", "#output-connector", "#input-connector"]) {
    document.querySelector(selector)?.classList.add("hidden");
  }
  // The idle and header copy both tell you to send a message -- but the chat
  // box is one of the panels just hidden, so in offline mode they have to say
  // what is actually true of a static, GPU-less page.
  byId("graph-idle").textContent = params.get("trace")
    ? "Loading a real sealed capture from an NVIDIA DGX Spark…"
    : "Click “Load sample trace” above to open a real capture from an NVIDIA DGX Spark — no GPU needed.";
  byId("run-subtitle").textContent = "Sealed trace viewer — real captured data, no GPU required.";
} else if (replayUrl) {
  // Replay stands in for the transport, so it has to be installed before any
  // socket is opened -- connectWebSocket() and connectCuptiActivity() capture
  // window.WebSocket at call time. Nothing downstream knows the difference.
  state.livePage = true;
  state.enginePaused = null; // a recording has no engine to pause
  installReplay(replayUrl).then((player) => {
    state.replay = player;
    // Seeking backwards replays from the start, so everything that accumulates
    // has to be cleared first: the trace, the CUPTI classifier, and the text.
    player.onReset = () => {
      ensureLiveTrace();
      resetCuptiAll();
      state.trace.query.request.messages = [
        { role: "user", content: player.manifest?.prompt ?? "(recorded prompt)" },
      ];
      byId("chat-reply").textContent = "";
      setThinkingPhase(false);
    };
    player.onChatDelta = (piece) => {
      const reply = byId("chat-reply");
      reply.textContent += piece;
      if (piece.includes("<think>")) setThinkingPhase(true);
      if (piece.includes("</think>")) setThinkingPhase(false);
    };
    const input = byId("chat-input");
    if (input) {
      // Never prefill the box and never disable it. Disabling it left anyone
      // whose URL still carried ?replay= -- from a reload, a bookmark or a
      // restored tab -- staring at a chat box that silently refused to take a
      // question, with no way to tell why. Typing one here now leaves the
      // recording and asks it live (see the submit handler).
      input.value = "";
      input.placeholder = "Ask your own question — this leaves the recording";
    }
      showReplayMarker(player);
    byId("run-subtitle").textContent =
      `Replay of a capture recorded on ${player.manifest?.host ?? "a DGX Spark"} — real measured data, reproduced timing.`;
    byId("graph-idle").textContent =
      "Press \u25B6 to play the recording. Set the speed to .02x first to watch one step's kernel sweep unfold.";
    updateTransportEnabled();
    updatePlayButton();
    connectWebSocket();
    connectCuptiActivity();
    renderSpine();
  }).catch((error) => {
    byId("error-panel").textContent = `Could not load the replay capture: ${error.message}`;
    byId("error-panel").classList.remove("hidden");
  });
} else {
  state.livePage = true;
  connectWebSocket();
  syncEnginePaused();
  // Sent a moment late on purpose: the semantic socket has to be open before
  // the request is admitted, or the first engine steps arrive with nothing
  // listening and the recording starts mid-answer.
  let carried = null;
  try {
    carried = sessionStorage.getItem(PENDING_PROMPT_KEY);
    sessionStorage.removeItem(PENDING_PROMPT_KEY);
  } catch (error) {
    carried = null; // blocked storage: nothing was carried
  }
  if (carried) setTimeout(() => sendChatMessage(carried), 600);
  if (shadowEnabled) {
    document.querySelector(".kernel-activity-card")?.classList.remove("hidden");
    connectKernelActivity();
  }
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
  loadUrl("../ui/data/sample-qwen3-14b-trace-v2.json")
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

