const REQUEST_COLORS = [
  "#72e3b1", "#64c7e8", "#ffc66d", "#b89cff",
  "#ff8b7a", "#77a7ff", "#e997c5", "#9bd36a",
];

const SVG_NS = "http://www.w3.org/2000/svg";
const state = { data: null, requestId: null, stepId: null };
const byId = (id) => document.getElementById(id);
const colorFor = (requestIndex) => REQUEST_COLORS[requestIndex % REQUEST_COLORS.length];
const escapeMarkup = (value) => String(value)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;");

function requestById(id) {
  return state.data.requests.find((request) => request.id === id);
}

function currentRequest() {
  return requestById(state.requestId);
}

function currentStep() {
  return state.data.steps.find((step) => step.id === state.stepId);
}

function rowForRequest(step, requestId) {
  return step.packedRows.find((row) => row.requestId === requestId);
}

function centerStart(count, height = 34, gap = 13) {
  const total = count * height + Math.max(0, count - 1) * gap;
  return (560 - total) / 2;
}

function curve(x1, y1, x2, y2) {
  const bend = (x2 - x1) * .47;
  return `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`;
}

function requestEntries(step) {
  return step.packedRows.map((row) => ({
    ...row,
    request: requestById(row.requestId),
    color: colorFor(row.requestIndex),
    padding: false,
  }));
}

function packedEntries(step) {
  const entries = requestEntries(step);
  for (let index = 0; index < step.paddingRows; index += 1) {
    entries.push({
      requestId: null,
      label: "Graph padding",
      request: null,
      color: "#60726b",
      padding: true,
      rowBegin: step.scheduledTokens + index,
      rowEnd: step.scheduledTokens + index + 1,
      tokens: 0,
      schedulerPosition: null,
    });
  }
  return entries;
}

function nodeMarkup({ className, x, y, width, height, label, detail, index, color, selected, padding, requestId }) {
  return `
    <g class="${className} ${selected ? "selected" : ""} ${padding ? "padding" : ""}"
       data-request="${requestId ?? ""}" style="--node-color:${color}">
      <rect class="node-rect" x="${x}" y="${y}" width="${width}" height="${height}"></rect>
      <circle class="node-swatch" cx="${x + 12}" cy="${y + height / 2}" r="3"></circle>
      <text class="node-label" x="${x + 22}" y="${y + 14}">${escapeMarkup(label)}</text>
      <text class="node-detail" x="${x + 22}" y="${y + 26}">${escapeMarkup(detail)}</text>
      <text class="node-index" x="${x + width - 10}" y="${y + 20}" text-anchor="end">${escapeMarkup(index)}</text>
    </g>
  `;
}

function renderGraph() {
  const step = currentStep();
  const selectedRequest = currentRequest();
  const scheduler = step.schedulerOrder.map((entry) => {
    const request = requestById(entry.requestId);
    return {
      ...entry,
      request,
      color: colorFor(request.colorIndex),
      padding: false,
    };
  });
  const packed = packedEntries(step);
  const nodeHeight = 34;
  const nodeGap = 13;
  const schedulerStart = centerStart(scheduler.length, nodeHeight, nodeGap);
  const packedStart = centerStart(packed.length, nodeHeight, nodeGap);
  const nodeY = (start, index) => start + index * (nodeHeight + nodeGap);
  const packedIndex = new Map(packed.filter((entry) => !entry.padding)
    .map((entry, index) => [entry.requestId, index]));
  const schedulerIndex = new Map(scheduler.map((entry, index) => [entry.requestId, index]));
  const x = { scheduler: 24, packed: 338, kernel: 620, owner: 1022 };
  const width = { scheduler: 150, packed: 150, kernel: 255, owner: 154 };
  const shellY = 24;
  const shellHeight = 512;
  const selectedId = selectedRequest.id;

  const schedulerLinks = scheduler.map((entry, index) => {
    const targetIndex = packedIndex.get(entry.requestId);
    const y1 = nodeY(schedulerStart, index) + nodeHeight / 2;
    const y2 = nodeY(packedStart, targetIndex) + nodeHeight / 2;
    const selected = entry.requestId === selectedId;
    return `<path class="flow-link measured ${selected ? "selected" : ""}" style="--link-color:${entry.color}" stroke="${entry.color}" d="${curve(x.scheduler + width.scheduler, y1, x.packed, y2)}"></path>`;
  }).join("");

  const sharedLinks = packed.map((entry, index) => {
    const y = nodeY(packedStart, index) + nodeHeight / 2;
    const selected = entry.requestId === selectedId;
    return `<path class="flow-link shared ${selected ? "selected" : ""}" style="--link-color:${entry.color}" stroke="${entry.color}" d="${curve(x.packed + width.packed, y, x.kernel, y)}"></path>`;
  }).join("");

  const ownershipLinks = packed.map((entry, index) => {
    const y = nodeY(packedStart, index) + nodeHeight / 2;
    const selected = entry.requestId === selectedId;
    return `<path class="flow-link inferred ${selected ? "selected" : ""}" style="--link-color:${entry.color}" stroke="${entry.color}" d="${curve(x.kernel + width.kernel, y, x.owner, y)}"></path>`;
  }).join("");

  const schedulerNodes = scheduler.map((entry, index) => nodeMarkup({
    className: "flow-request-node",
    x: x.scheduler,
    y: nodeY(schedulerStart, index),
    width: width.scheduler,
    height: nodeHeight,
    label: entry.request.label,
    detail: `${entry.tokens} token${entry.tokens === 1 ? "" : "s"}`,
    index: `pos ${index}`,
    color: entry.color,
    selected: entry.requestId === selectedId,
    padding: false,
    requestId: entry.requestId,
  })).join("");

  const packedNodes = packed.map((entry, index) => nodeMarkup({
    className: "flow-packed-node",
    x: x.packed,
    y: nodeY(packedStart, index),
    width: width.packed,
    height: nodeHeight,
    label: entry.padding ? "Graph padding" : entry.request.label,
    detail: entry.padding ? "no request owner" : `rows [${entry.rowBegin},${entry.rowEnd})`,
    index: `row ${entry.rowBegin}`,
    color: entry.color,
    selected: entry.requestId === selectedId,
    padding: entry.padding,
    requestId: entry.requestId,
  })).join("");

  const ownerNodes = packed.map((entry, index) => {
    const blocks = entry.padding
      ? step.kernels.length
      : step.kernels.length * (entry.rowEnd - entry.rowBegin);
    return nodeMarkup({
      className: "flow-owner-node",
      x: x.owner,
      y: nodeY(packedStart, index),
      width: width.owner,
      height: nodeHeight,
      label: entry.padding ? "Padding work" : entry.request.label,
      detail: `${blocks} block${blocks === 1 ? "" : "s"}`,
      index: entry.padding ? "discard" : `idx ${entry.rowBegin}`,
      color: entry.color,
      selected: entry.requestId === selectedId,
      padding: entry.padding,
      requestId: entry.requestId,
    });
  }).join("");

  const bands = packed.map((entry, index) => {
    const y = nodeY(packedStart, index) + nodeHeight / 2 - 5;
    return `<rect class="kernel-band ${entry.requestId === selectedId ? "selected" : ""}" style="--band-color:${entry.color}" x="${x.kernel + 14}" y="${y}" width="${width.kernel - 28}" height="10" rx="5"></rect>`;
  }).join("");

  const kernelColumns = step.kernels.map((kernel, index) => {
    const available = width.kernel - 32;
    const columnWidth = Math.max(2, available / step.kernels.length - 1.3);
    const columnX = x.kernel + 16 + index * (available / step.kernels.length);
    return `
      <rect class="kernel-layer" x="${columnX.toFixed(2)}" y="121" width="${columnWidth.toFixed(2)}" height="354">
        <title>Kernel ${index}: ${kernel.durationUs} µs, correlation ${kernel.correlation}</title>
      </rect>
    `;
  }).join("");

  const reordered = step.reorderedPositions > 0 ? `
    <g class="reorder-callout">
      <rect x="205" y="37" width="104" height="26"></rect>
      <text x="257" y="53">${step.reorderedPositions} positions moved</text>
    </g>
  ` : "";

  const selectedPacked = rowForRequest(step, selectedId);
  const selectedSchedulerPosition = schedulerIndex.get(selectedId);
  const selectedPackedPosition = packedIndex.get(selectedId);
  const crossing = selectedSchedulerPosition !== selectedPackedPosition;
  const movementLabel = crossing
    ? `selected path crosses: scheduler ${selectedSchedulerPosition} → packed row ${selectedPacked.rowBegin}`
    : `selected path stays aligned at row ${selectedPacked.rowBegin}`;

  byId("flow-graph").innerHTML = `
    <g aria-label="Measured scheduler-to-packed-row links">${schedulerLinks}</g>
    <g aria-label="Measured packed rows entering shared GPU kernels">${sharedLinks}</g>
    <g aria-label="Reconstructed kernel-block ownership links">${ownershipLinks}</g>
    ${reordered}
    <text class="column-note" x="24" y="535">${scheduler.length} scheduled requests</text>
    <text class="column-note" x="338" y="535">${movementLabel}</text>
    ${schedulerNodes}
    ${packedNodes}
    <g class="kernel-group">
      <rect class="kernel-shell" x="${x.kernel}" y="${shellY}" width="${width.kernel}" height="${shellHeight}"></rect>
      <text class="kernel-title" x="${x.kernel + 17}" y="54">${step.kernels.length} cache-kernel launches</text>
      <text class="kernel-subtitle" x="${x.kernel + 17}" y="71">stream ${step.kernels[0]?.stream ?? "—"} · graph ${step.kernels[0]?.graphId ?? "—"} · grid.x ${step.gridX}</text>
      <text class="kernel-subtitle" x="${x.kernel + 17}" y="99">K0</text>
      <text class="kernel-subtitle" x="${x.kernel + width.kernel - 17}" y="99" text-anchor="end">K${Math.max(0, step.kernels.length - 1)}</text>
      ${bands}
      ${kernelColumns}
      <text class="column-note" x="${x.kernel + 17}" y="512">each vertical strip is one timed kernel</text>
    </g>
    ${ownerNodes}
  `;

  byId("flow-graph").querySelectorAll("[data-request]").forEach((node) => {
    if (!node.dataset.request) return;
    node.addEventListener("click", () => {
      state.requestId = node.dataset.request;
      renderControls();
      render();
    });
  });
}

function renderControls() {
  const request = currentRequest();
  const requestSelect = byId("request-select");
  requestSelect.innerHTML = state.data.requests.map((entry) =>
    `<option value="${entry.id}" ${entry.id === request.id ? "selected" : ""}>${escapeMarkup(entry.label)} · ${entry.id}</option>`
  ).join("");

  const availableSteps = state.data.steps.filter((step) => step.phase === "decode" && request.stepIds.includes(step.id));
  if (!availableSteps.some((step) => step.id === state.stepId)) {
    state.stepId = availableSteps.find((step) => step.reorderedPositions > 0)?.id ?? availableSteps[0].id;
  }
  const stepSelect = byId("step-select");
  stepSelect.innerHTML = availableSteps.map((step) =>
    `<option value="${step.id}" ${step.id === state.stepId ? "selected" : ""}>Step ${step.id} · ${step.phase} · ${step.packedRows.length} requests</option>`
  ).join("");
}

function renderStory() {
  const step = currentStep();
  const request = currentRequest();
  const row = rowForRequest(step, request.id);
  const moved = row.schedulerPosition !== row.rowBegin;
  byId("story-title").textContent = moved
    ? `${request.label} moved from scheduler position ${row.schedulerPosition} to packed row ${row.rowBegin}.`
    : `${request.label} remained aligned at packed row ${row.rowBegin}.`;
  byId("story-detail").textContent =
    `Then ${step.kernels.length} cache kernels processed the shared ${step.gridX}-row CUDA Graph shape. Because blockIdx.x follows the authoritative packed row for this kernel family, row ${row.rowBegin} contributes ${step.kernels.length * (row.rowEnd - row.rowBegin)} attributed blocks to ${request.label}.`;
  const metrics = [
    [step.packedRows.length, "real requests"],
    [step.kernels.length, "kernel launches"],
    [step.gridX, "grid.x rows"],
    [step.paddingRows, "padding rows"],
  ];
  byId("story-metrics").innerHTML = metrics.map(([value, label]) =>
    `<div class="story-metric"><strong>${value}</strong><small>${label}</small></div>`
  ).join("");

  const range = row.rowEnd - row.rowBegin === 1
    ? `blockIdx.x = ${row.rowBegin}`
    : `blockIdx.x ∈ [${row.rowBegin}, ${row.rowEnd})`;
  byId("ownership-formula").innerHTML =
    `<span class="accent">${range}</span> → packed rows [${row.rowBegin},${row.rowEnd}) → ${escapeMarkup(request.label)}`;
  byId("formula-explanation").textContent =
    "The first two arrows use measured semantic and launch data. Request ownership is the final arithmetic reconstruction; it is not presented as a same-run device callback.";
}

function render() {
  renderStory();
  renderGraph();
}

async function boot() {
  const response = await fetch("./data/exp0020.json");
  if (!response.ok) throw new Error(`trace fetch failed: HTTP ${response.status}`);
  const data = await response.json();
  if (data.schema !== "GPU_OBSERVER_UI_01") throw new Error(`unsupported schema: ${data.schema}`);
  state.data = data;
  state.requestId = data.defaults.requestId;
  state.stepId = data.defaults.stepId;
  renderControls();
  render();

  byId("request-select").addEventListener("change", (event) => {
    state.requestId = event.target.value;
    renderControls();
    render();
  });
  byId("step-select").addEventListener("change", (event) => {
    state.stepId = Number(event.target.value);
    render();
  });
}

boot().catch((error) => {
  document.querySelector(".flow-app").innerHTML =
    `<section class="story-card"><div><h1>Graph could not load</h1><p>${escapeMarkup(error.message)}</p></div></section>`;
});
