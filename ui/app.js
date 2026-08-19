const REQUEST_COLORS = [
  "#72e3b1", "#64c7e8", "#ffc66d", "#b89cff",
  "#ff8b7a", "#77a7ff", "#e997c5", "#9bd36a",
];

const state = { data: null, requestId: null, stepId: null };
const byId = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");
const formatMs = (value) => `${Number(value).toFixed(value < 1 ? 3 : 2)} ms`;
const requestColor = (request) => REQUEST_COLORS[request.colorIndex % REQUEST_COLORS.length];

function currentRequest() {
  return state.data.requests.find((request) => request.id === state.requestId);
}

function currentStep() {
  return state.data.steps.find((step) => step.id === state.stepId);
}

function selectedRow(step, requestId) {
  return step.packedRows.find((row) => row.requestId === requestId);
}

function setRequest(requestId) {
  state.requestId = requestId;
  const request = currentRequest();
  if (!request.stepIds.includes(state.stepId)) {
    const interesting = state.data.steps.find((step) =>
      request.stepIds.includes(step.id) && step.reorderedPositions > 0);
    state.stepId = interesting?.id ?? request.stepIds[0];
  }
  render();
}

function setStep(stepId) {
  state.stepId = stepId;
  render();
}

function renderSidebar() {
  const { experiment, requests } = state.data;
  byId("run-result").textContent = experiment.result;
  byId("run-id").textContent = experiment.id;
  byId("run-model").textContent = experiment.model;
  byId("run-mode").textContent = experiment.executionMode;
  byId("request-count").textContent = requests.length;
  byId("request-list").innerHTML = requests.map((request) => `
    <button class="request-button ${request.id === state.requestId ? "active" : ""}"
      data-request="${escapeHtml(request.id)}" style="--request-color:${requestColor(request)}">
      <span class="request-swatch"></span>
      <span><strong>${escapeHtml(request.label)}</strong><br><code>${escapeHtml(request.id)}</code></span>
      <small>${request.observedSteps} steps</small>
    </button>
  `).join("");
  byId("request-list").querySelectorAll("button").forEach((button) =>
    button.addEventListener("click", () => setRequest(button.dataset.request)));
}

function renderHeader() {
  const { experiment } = state.data;
  const request = currentRequest();
  const step = currentStep();
  byId("page-title").textContent = `${request.label} through engine step ${step.id}`;
  byId("quality-pills").innerHTML = [
    `<span class="quality-pill"><strong>${experiment.semanticLoss}</strong> semantic loss</span>`,
    `<span class="quality-pill"><strong>${experiment.clockStartUncertaintyNs} ns</strong> clock uncertainty</span>`,
    `<span class="quality-pill"><strong>${experiment.result}</strong> quality gate</span>`,
  ].join("");
}

function renderHero() {
  const request = currentRequest();
  const step = currentStep();
  const row = selectedRow(step, request.id);
  byId("hero-title").textContent = `${request.label} is row ${row.rowBegin} in step ${step.id}`;
  byId("hero-description").textContent = step.reorderedPositions
    ? `The Python scheduler selected this request at list position ${row.schedulerPosition}. GPUModelRunner compacted the persistent batch and made it packed row ${row.rowBegin}; that authoritative row—not scheduler order—owns one block in each selected cache kernel.`
    : "Scheduler and packed order agree in this step. The authoritative packed row still supplies ownership because later compaction can change the physical tensor position.";
  const metrics = [
    ["Step wall time", formatMs(step.wallMs), "vLLM semantic boundary"],
    ["Target kernels", step.kernels.length, "one per transformer layer"],
    ["Attributed blocks", step.kernels.length * (row.rowEnd - row.rowBegin), "geometry reconstruction"],
    ["Packed rows", `${step.scheduledTokens}/${step.gridX}`, `${step.paddingRows} graph padding`],
  ];
  byId("metric-grid").innerHTML = metrics.map(([label, value, detail]) => `
    <article class="metric-card"><span>${label}</span><strong>${value}</strong><small>${detail}</small></article>
  `).join("");
}

function renderCausalPath() {
  const request = currentRequest();
  const step = currentStep();
  const row = selectedRow(step, request.id);
  const color = requestColor(request);
  const nodes = [
    ["01 · request", request.label, `${request.id} · observed in ${request.observedSteps} steps`],
    ["02 · scheduler", `Step ${step.id} · ${step.phase}`, `position ${row.schedulerPosition} · ${step.scheduledTokens} scheduled tokens`],
    ["03 · model runner", `Packed row ${row.rowBegin}`, step.reorderedPositions ? `compacted · ${step.reorderedPositions} positions changed` : "stable layout"],
    ["04 · GPU", `${step.kernels.length} cache kernels`, `stream ${step.kernels[0]?.stream ?? "—"} · CUDA Graph ${step.kernels[0]?.graphId ?? "—"}`],
    ["05 · ownership", `${step.kernels.length} row blocks`, "arithmetic attribution · not a SASS callback"],
  ];
  byId("causal-path").innerHTML = nodes.map(([index, title, detail], nodeIndex) => `
    <article class="path-node ${nodeIndex === 0 || nodeIndex === 2 ? "selected" : ""}" style="--request-color:${color}">
      <span class="node-index">${index}</span><strong>${escapeHtml(title)}</strong><small>${escapeHtml(detail)}</small>
    </article>
  `).join("");
}

function renderStepRail() {
  const request = currentRequest();
  const steps = state.data.steps.filter((step) => request.stepIds.includes(step.id));
  byId("step-rail").innerHTML = steps.map((step) => `
    <button class="step-button ${step.phase} ${step.id === state.stepId ? "active" : ""} ${step.reorderedPositions ? "reordered" : ""}"
      data-step="${step.id}" role="option" aria-selected="${step.id === state.stepId}">
      <strong>${step.id}</strong><small>${step.scheduledTokens}r</small>
    </button>
  `).join("");
  byId("step-rail").querySelectorAll("button").forEach((button) =>
    button.addEventListener("click", () => setStep(Number(button.dataset.step))));
  byId("step-rail").querySelector(".active")?.scrollIntoView({ inline: "center", block: "nearest" });
}

function positionPercent(value, wallMs) {
  return Math.max(0, Math.min(100, (value / wallMs) * 100));
}

function renderTimeline() {
  const step = currentStep();
  byId("timeline-title").textContent = `Step ${step.id} · ${step.phase}`;
  const kernelTicks = step.kernels.map((kernel) => {
    const left = positionPercent(kernel.startMs, step.wallMs);
    const width = Math.max(0.16, positionPercent(kernel.endMs - kernel.startMs, step.wallMs));
    return `<span class="kernel-tick" style="left:${left}%;width:${width}%" title="Layer ${kernel.layerOrdinal}: ${kernel.durationUs} µs"></span>`;
  }).join("");
  const submitTicks = step.kernels.map((kernel) =>
    `<span class="submit-tick" style="left:${positionPercent(kernel.runtimeSubmitMs, step.wallMs)}%" title="Correlation ${kernel.correlation}"></span>`).join("");
  const packedPosition = positionPercent(step.packedAtMs, step.wallMs);
  byId("timeline").innerHTML = `
    <div class="timeline-row">
      <div class="timeline-label"><strong>Engine step</strong><small>Python begin → end</small></div>
      <div class="timeline-track"><span class="host-span"></span><span class="packed-marker" style="left:${packedPosition}%" title="Packed layout emitted"></span></div>
      <span class="timeline-value">${formatMs(step.wallMs)}</span>
    </div>
    <div class="timeline-row">
      <div class="timeline-label"><strong>Submissions</strong><small>CUPTI runtime API</small></div>
      <div class="timeline-track">${submitTicks}</div>
      <span class="timeline-value">${step.runtimeSubmitSpanMs == null ? "—" : formatMs(step.runtimeSubmitSpanMs)}</span>
    </div>
    <div class="timeline-row">
      <div class="timeline-label"><strong>GPU cache kernels</strong><small>actual start → end</small></div>
      <div class="timeline-track">${kernelTicks}</div>
      <span class="timeline-value">Σ ${(step.targetKernelSumUs / 1000).toFixed(3)} ms</span>
    </div>
    <div class="timeline-axis"><span>0 ms</span><span>${(step.wallMs * .25).toFixed(0)}</span><span>${(step.wallMs * .5).toFixed(0)}</span><span>${(step.wallMs * .75).toFixed(0)}</span><span>${step.wallMs.toFixed(0)} ms</span></div>
    <div class="timeline-warning">Dark space is <strong>unexplained by this target-filtered trace</strong>, not proven GPU idle time. Only <code>reshape_and_cache_flash_kernel</code> intervals were retained.</div>
  `;
}

function renderPacking() {
  const request = currentRequest();
  const step = currentStep();
  const color = requestColor(request);
  const stable = step.reorderedPositions === 0;
  const badge = byId("reorder-badge");
  badge.textContent = stable ? "Order stable" : `${step.reorderedPositions} positions changed`;
  badge.classList.toggle("stable", stable);
  const scheduler = step.schedulerOrder.map((row) => `
    <div class="packing-row ${row.requestId === request.id ? "selected" : ""}" style="--request-color:${color}">
      <span class="row-index">${row.position}</span><span>${escapeHtml(row.label)}</span>
    </div>
  `).join("");
  const packed = [
    ...step.packedRows.map((row) => `
      <div class="packing-row ${row.requestId === request.id ? "selected" : ""}" style="--request-color:${color}">
        <span class="row-index">${row.rowBegin}</span><span>${escapeHtml(row.label)}</span>
      </div>
    `),
    ...Array.from({ length: step.paddingRows }, (_, index) => `
      <div class="packing-row padding"><span class="row-index">${step.scheduledTokens + index}</span><span>Graph padding</span></div>
    `),
  ].join("");
  byId("packing-comparison").innerHTML = `
    <div class="packing-column"><h4>Scheduler list position</h4><div class="packing-list">${scheduler}</div></div>
    <div class="packing-arrow">→</div>
    <div class="packing-column"><h4>Packed tensor row</h4><div class="packing-list">${packed}</div></div>
  `;
}

function renderKernels() {
  const step = currentStep();
  byId("kernel-family").textContent = step.targetKernelFamily;
  byId("kernel-grid").innerHTML = step.kernels.map((kernel) => `
    <div class="kernel-cell" title="Layer ordinal ${kernel.layerOrdinal}\n${kernel.durationUs} µs\ngrid.x=${kernel.gridX}, block.x=${kernel.blockX}\ncorrelation=${kernel.correlation}">
      ${String(kernel.layerOrdinal).padStart(2, "0")}
    </div>
  `).join("");
}

function renderEvidence() {
  const groups = [
    ["measured", "Measured", state.data.evidence.measured],
    ["inferred", "Reconstructed", state.data.evidence.inferred],
    ["unavailable", "Not captured", state.data.evidence.unavailable],
  ];
  byId("evidence-grid").innerHTML = groups.map(([className, title, items]) => `
    <article class="evidence-card ${className}"><h3>${title}</h3><ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></article>
  `).join("");
  byId("artifact-path").textContent = state.data.experiment.rawArtifact;
}

function render() {
  renderSidebar();
  renderHeader();
  renderHero();
  renderCausalPath();
  renderStepRail();
  renderTimeline();
  renderPacking();
  renderKernels();
  renderEvidence();
}

async function boot() {
  try {
    const response = await fetch("./data/exp0020.json");
    if (!response.ok) throw new Error(`trace fetch failed: HTTP ${response.status}`);
    const data = await response.json();
    if (data.schema !== "GPU_OBSERVER_UI_01") throw new Error(`unsupported schema ${data.schema}`);
    state.data = data;
    state.requestId = data.defaults.requestId;
    state.stepId = data.defaults.stepId;
    render();
  } catch (error) {
    document.querySelector("main").innerHTML = `<section class="panel"><h2>Trace could not load</h2><p>${escapeHtml(error.message)}</p><p>Serve this directory over HTTP; browsers block fetch from <code>file://</code>.</p></section>`;
  }
}

boot();
