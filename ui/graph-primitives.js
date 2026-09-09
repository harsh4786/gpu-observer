// Shared SVG node/edge drawing primitives, extracted from causal-graph.js so
// the same visual grammar (rounded node cards, bezier edges, lane headings,
// evidence-tinted markers) is reused verbatim by any other graph renderer in
// this UI -- currently causal-graph.js and kernel-graph.js -- instead of a
// second implementation drifting out of sync with it.

export const SVG_NS = "http://www.w3.org/2000/svg";

export function svgElement(name, attributes = {}) {
  const node = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

export function truncate(value, limit) {
  const text = String(value ?? "").replaceAll("\n", " ").replace(/\s+/g, " ").trim();
  return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
}

// Stable, deterministic per-request color from a hash of the request id --
// same request always gets the same color within a page load, with no
// server-side coordination needed. Shared so any view attributing evidence
// to a request (causal-graph.js's lanes, kernel-activity.js's SM grid) uses
// one consistent palette instead of each inventing its own.
export function requestColor(requestId) {
  const colors = ["#72e3b1", "#64c7e8", "#ffc66d", "#b89cff", "#ff8b7a", "#77a7ff"];
  let hash = 0;
  for (const char of String(requestId)) hash = (hash * 33 + char.charCodeAt(0)) >>> 0;
  return colors[hash % colors.length];
}

export function addMarker(defs, id, color) {
  const marker = svgElement("marker", {
    id,
    viewBox: "0 0 10 10",
    refX: 9,
    refY: 5,
    markerWidth: 6,
    markerHeight: 6,
    orient: "auto-start-reverse",
  });
  marker.append(svgElement("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: color }));
  defs.append(marker);
}

export function addHeading(svg, x, title, subtitle) {
  const titleNode = svgElement("text", { x, y: 27, class: "graph-lane-title" });
  titleNode.textContent = title;
  const subtitleNode = svgElement("text", { x, y: 43, class: "graph-lane-subtitle" });
  subtitleNode.textContent = subtitle;
  svg.append(titleNode, subtitleNode);
}

export function addNode(svg, options) {
  const {
    x, y, width = 176, height = 58, color = "#60756d", kind = "measured",
    title, titleLimit = 26, lines = [], badge, focus = false, pulse = false, onActivate, tooltip,
  } = options;
  const group = svgElement("g", {
    class: `graph-node ${kind}${focus ? " focus" : ""}${pulse ? " live-active" : ""}${onActivate ? " interactive" : ""}`,
    transform: `translate(${x} ${y})`,
  });
  if (onActivate) {
    group.setAttribute("role", "button");
    group.setAttribute("tabindex", "0");
    group.addEventListener("click", onActivate);
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        onActivate();
      }
    });
  }
  group.style.setProperty("--node-color", color);
  if (tooltip) {
    const titleTip = svgElement("title");
    titleTip.textContent = tooltip;
    group.append(titleTip);
  }
  group.append(svgElement("rect", { width, height, rx: 10 }));

  // Character budgets derived from the node's actual width instead of fixed
  // limits: the monospace advances below match trace.css's .graph-node-*
  // rules (13.5px title, 11px line, 9px badge). Fixed limits let a 30-char
  // line spill past a 168px node -- visible on the offline graph as the
  // query text running under the "tokenize" edge label.
  const PAD_X = 23;
  const badgeWidth = badge ? String(badge).length * 5.4 + 10 : 0;
  const titleBudget = Math.max(6, Math.floor((width - PAD_X - badgeWidth) / 8.1));
  const lineBudget = Math.max(6, Math.floor((width - PAD_X) / 6.6));

  // Only a node with NO lines gets a vertically centered title. Centering it
  // for a one-line node put the title's baseline at height/2+5 while the line
  // stayed pinned at 39 -- for the 68px-tall GPU node those are both y=39, so
  // the kernel name rendered directly on top of "N kernels".
  const titleOnly = lines.length === 0;
  const titleNode = svgElement("text", {
    x: 13,
    y: titleOnly ? Math.round(height / 2) + 5 : 22,
    class: "graph-node-title",
  });
  titleNode.textContent = truncate(title, Math.min(titleLimit, titleBudget));
  group.append(titleNode);
  lines.slice(0, 2).forEach((line, index) => {
    const lineNode = svgElement("text", { x: 13, y: 39 + index * 16, class: "graph-node-line" });
    lineNode.textContent = truncate(line, lineBudget);
    group.append(lineNode);
  });
  if (badge) {
    const badgeNode = svgElement("text", {
      x: width - 9,
      y: 18,
      class: "graph-node-badge",
      "text-anchor": "end",
    });
    badgeNode.textContent = badge;
    group.append(badgeNode);
  }
  svg.append(group);
  return {
    left: x,
    right: x + width,
    top: y,
    bottom: y + height,
    cx: x + width / 2,
    cy: y + height / 2,
  };
}

export function addEdge(svg, from, to, options = {}) {
  const {
    color = "#72e3b1", marker = "arrow-measured", width = 2,
    kind = "measured", label = "", dashed = false, flowing = false,
  } = options;
  const bend = (from.right + to.left) / 2;
  const path = svgElement("path", {
    d: `M ${from.right} ${from.cy} C ${bend} ${from.cy}, ${bend} ${to.cy}, ${to.left} ${to.cy}`,
    class: `graph-edge ${kind}${flowing ? " flowing" : ""}`,
    stroke: color,
    "stroke-width": Math.max(1.2, width),
    "marker-end": `url(#${marker})`,
  });
  if (dashed && !flowing) path.setAttribute("stroke-dasharray", "7 6");
  svg.insertBefore(path, svg.querySelector(".graph-node"));
  // Edge labels used to sit at min(cy) - 8, which is *inside* the vertical
  // span of both nodes: with only 24px between the query and tokenizer nodes,
  // "tokenize" rendered straight across both borders. Prefer the gap when the
  // label fits there, else lift it clear above the nodes, else drop it --
  // never draw a label on top of a node. y=43 is the lane subtitle baseline,
  // so anything above ~50 would collide with the column headings instead.
  if (label) {
    const gap = to.left - from.right;
    const labelWidth = String(label).length * 6 + 8;
    const above = Math.min(from.top, to.top) - 7;
    let y = null;
    if (gap >= labelWidth) y = Math.min(from.cy, to.cy) - 8;
    else if (above >= 50) y = above;
    if (y !== null) {
      const text = svgElement("text", {
        x: bend,
        y,
        class: `graph-edge-label ${kind}`,
        "text-anchor": "middle",
      });
      text.textContent = label;
      svg.append(text);
    }
  }
}

export function scrollToCard(id) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "center" });
}

/**
 * A slim horizontal progress bar -- track rect + fill rect scaled by
 * `fraction`. Simpler than addProgressRing's radial dasharray trick; used
 * where a lane needs a compact linear "N of total" readout rather than a
 * per-node ring.
 */
export function addProgressBar(svg, { x, y, width, height = 5, fraction = 0, color = "#64c7e8" }) {
  const group = svgElement("g", { class: "progress-bar" });
  group.append(svgElement("rect", { x, y, width, height, rx: height / 2, class: "progress-bar-track" }));
  const fillWidth = Math.max(0, Math.min(1, fraction)) * width;
  if (fillWidth > 0) {
    group.append(svgElement("rect", {
      x, y, width: fillWidth, height, rx: height / 2,
      class: "progress-bar-fill", fill: color,
    }));
  }
  svg.append(group);
  return group;
}

/**
 * A small stepper chip: compact rounded square + short abbreviated label,
 * with a native <title> child for hover discoverability of the full name.
 * Deliberately much smaller than addNode's card (title + up to 2 lines) --
 * built for laying 11 of these in one row without the lane dominating the
 * whole graph. Returns the same anchor-geometry shape addNode does so
 * addEdge can target a chip directly.
 */
export function addStageChip(svg, options) {
  const {
    x, y, size = 18, color = "#60756d", kind = "illustrative",
    label, title, active = false, onActivate,
  } = options;
  const group = svgElement("g", {
    class: `stage-chip ${kind}${active ? " active" : ""}${onActivate ? " interactive" : ""}`,
    transform: `translate(${x} ${y})`,
  });
  if (onActivate) {
    group.setAttribute("role", "button");
    group.setAttribute("tabindex", "0");
    group.addEventListener("click", onActivate);
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        onActivate();
      }
    });
  }
  group.style.setProperty("--node-color", color);
  const titleNode = svgElement("title");
  titleNode.textContent = title ?? label;
  group.append(titleNode);
  group.append(svgElement("rect", { width: size, height: size, rx: 4 }));
  const labelNode = svgElement("text", {
    x: size / 2, y: size / 2 + 3, "text-anchor": "middle", class: "stage-chip-label",
  });
  labelNode.textContent = label;
  group.append(labelNode);
  svg.append(group);
  return {
    left: x, right: x + size, top: y, bottom: y + size,
    cx: x + size / 2, cy: y + size / 2,
  };
}

/**
 * A compact radial progress indicator (classic stroke-dasharray trick: a
 * full-circumference dash with a dashoffset that reveals `fraction` of it).
 * Used to fold what used to be a whole separate card's worth of "where are
 * we in this sequence" into a small always-visible element on a live node,
 * instead of a big separate rail the user has to go look at.
 */
export function addProgressRing(svg, { cx, cy, radius = 15, fraction = 0, color = "#64c7e8", label = "" }) {
  const group = svgElement("g", { class: "progress-ring" });
  const circumference = 2 * Math.PI * radius;
  group.append(svgElement("circle", { cx, cy, r: radius, class: "progress-ring-track" }));
  group.append(svgElement("circle", {
    cx, cy, r: radius,
    class: "progress-ring-arc",
    stroke: color,
    "stroke-dasharray": circumference.toFixed(2),
    "stroke-dashoffset": (circumference * (1 - Math.max(0, Math.min(1, fraction)))).toFixed(2),
    transform: `rotate(-90 ${cx} ${cy})`,
  }));
  if (label) {
    const text = svgElement("text", { x: cx, y: cy + 3, class: "progress-ring-label", "text-anchor": "middle" });
    text.textContent = label;
    group.append(text);
  }
  svg.append(group);
}
