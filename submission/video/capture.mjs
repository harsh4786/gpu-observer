import { writeFileSync, readFileSync } from "node:fs";
const OUT = process.argv[2], REPO = process.argv[3];
const DRV = "http://127.0.0.1:4444";
const rq = async (m, p, b) => { const r = await fetch(DRV + p, { method: m, headers: { "Content-Type": "application/json" }, body: b ? JSON.stringify(b) : undefined }); const j = await r.json(); if (j.value && j.value.error) throw new Error(`${p}: ${j.value.message}`); return j.value; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const newSession = async (args, prefs = {}) => (await rq("POST", "/session", { capabilities: { alwaysMatch: { browserName: "firefox", "moz:firefoxOptions": { args, prefs, binary: "/snap/firefox/current/usr/lib/firefox/firefox" } } } })).sessionId;

// ---------- slides: scale each 1280x720 slide by 1.5 into a 1920x1080 frame ----------
{
  const sid = await newSession(["-headless", "--width=1920", "--height=1300"]);
  const ex = (x) => rq("POST", `/session/${sid}/execute/sync`, { script: x, args: [] });
  try {
    await rq("POST", `/session/${sid}/url`, { url: `file://${REPO}/submission/deck/deck.html` });
    await sleep(900);
    for (const n of [1, 2, 3, 4, 5, 7, 8, 9, 11, 12]) {
      await ex(`document.getElementById("vf")?.remove(); const s = document.querySelectorAll(".slide")[${n - 1}];
        const f = document.createElement("div"); f.id = "vf"; f.style.cssText = "position:fixed;left:0;top:0;width:1920px;height:1080px;overflow:hidden;z-index:9;background:#07100f";
        const c = s.cloneNode(true); c.style.cssText = "transform:scale(1.5);transform-origin:0 0;"; f.append(c); document.body.append(f); return 1;`);
      await sleep(250);
      const eid = Object.values(await rq("POST", `/session/${sid}/element`, { using: "css selector", value: "#vf" }))[0];
      writeFileSync(`${OUT}/slide-${n}.png`, Buffer.from(await rq("GET", `/session/${sid}/element/${eid}/screenshot`), "base64"));
    }
  } finally { await rq("DELETE", `/session/${sid}`).catch(() => {}); }
}

// ---------- viewer: 1536x864 CSS px at 1.25 device px => 1920x1080 frames ----------
const trace = JSON.parse(readFileSync(`${REPO}/ui/data/sample-qwen3-14b-trace-v2.json`, "utf8"));
const focus = trace.focus.requestHash;
const ids = (xs) => (xs || []).map((x) => x.requestId);
const multiIdx = trace.steps.findIndex((s) => (s.schedulerSlices || []).length >= 8 && ids(s.packedSlices).includes(focus));
const reorderIdx = trace.steps.findIndex((s) => { const a = ids(s.schedulerSlices), b = ids(s.packedSlices); return a.length > 1 && a.join() !== b.join() && [...a].sort().join() === [...b].sort().join(); });
const meta = { multiIdx, multiStep: trace.steps[multiIdx]?.id, reorderIdx, reorderStep: trace.steps[reorderIdx]?.id, reorderMismatches: trace.steps[reorderIdx]?.schedulerOrderMismatches };

const sid = await newSession(["-headless", "--width=1536", "--height=950"], { "layout.css.devPixelsPerPx": "1.25" });
const ex = (x) => rq("POST", `/session/${sid}/execute/sync`, { script: x, args: [] });
const rects = {};
const shot = async (name) => { await sleep(500); writeFileSync(`${OUT}/viewer-${name}.png`, Buffer.from(await rq("GET", `/session/${sid}/screenshot`), "base64")); };
const rectOf = (js) => ex(`const e = ${js}; if (!e) return null; const r = e.getBoundingClientRect(); return { x: (r.left + r.width / 2) * devicePixelRatio, y: (r.top + r.height / 2) * devicePixelRatio };`);
try {
  await rq("POST", `/session/${sid}/window/rect`, { width: 1536, height: 950 });
  await sleep(300);
  let inner = await ex(`return [innerWidth, innerHeight, devicePixelRatio];`);
  await rq("POST", `/session/${sid}/window/rect`, { width: 1536 + (1536 - inner[0]), height: 950 + (864 - inner[1]) });
  await sleep(300);
  meta.viewport = await ex(`return [innerWidth, innerHeight, devicePixelRatio];`);

  await rq("POST", `/session/${sid}/url`, { url: "https://harsh4786.github.io/gpu-observer/" });
  await sleep(3500);
  rects.loadSample = await rectOf(`document.querySelector("#load-sample")`);
  await shot("landing");

  await ex(`document.querySelector("#load-sample").click(); return 1;`);
  for (let i = 0; i < 40; i++) { const n = await ex(`return document.querySelector("#causal-graph").querySelectorAll("g").length;`); if (n > 5) break; await sleep(500); }
  await shot("loaded");

  const pick = async (idx, name) => {
    rects[`select_${name}`] = await rectOf(`document.querySelector("#graph-step-select")`);
    meta[`${name}Option`] = await ex(`const s = document.querySelector("#graph-step-select"); s.selectedIndex = ${idx}; s.dispatchEvent(new Event("change", { bubbles: true })); return s.options[${idx}]?.textContent;`);
    await ex(`document.querySelector(".graph-card").scrollIntoView({ block: "start" }); return 1;`);
    await shot(name);
  };
  await pick(multiIdx, "multilane");
  if (reorderIdx >= 0) await pick(reorderIdx, "reordered");

  // back to the focused multi-lane step, click the GPU node, show ownership
  await ex(`const s = document.querySelector("#graph-step-select"); s.selectedIndex = ${multiIdx}; s.dispatchEvent(new Event("change", { bubbles: true })); document.querySelector(".graph-card").scrollIntoView({ block: "start" }); return 1;`);
  await sleep(500);
  const gpuSel = `[...document.querySelectorAll("#causal-graph .graph-node")].find(g => /kernels?$/.test(g.querySelector(".graph-node-title")?.textContent || ""))`;
  rects.gpuNode = await rectOf(gpuSel);
  await shot("gpu-before-click");
  await ex(`const g = ${gpuSel}; g?.dispatchEvent(new MouseEvent("click", { bubbles: true })); return !!g;`);
  await sleep(400);
  await ex(`document.querySelector(".kernel-detail-card").scrollIntoView({ block: "center" }); return 1;`);
  await shot("ownership");
  await ex(`document.querySelector(".evidence-card").scrollIntoView({ block: "center" }); return 1;`);
  await shot("evidence");
} finally { await rq("DELETE", `/session/${sid}`).catch(() => {}); }
writeFileSync(`${OUT}/rects.json`, JSON.stringify({ rects, meta }, null, 2));
console.log(JSON.stringify({ rects, meta }, null, 2));
