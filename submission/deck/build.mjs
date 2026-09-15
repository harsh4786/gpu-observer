// Render submission/deck/deck.html to deck.pdf (12 x 16:9 pages) and cover.png
// (slide 1 at 1920x1080) through a running geckodriver on 127.0.0.1:4444.
// Usage: node submission/deck/build.mjs
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
const here = dirname(fileURLToPath(import.meta.url));
const DRV = "http://127.0.0.1:4444";
const rq = async (m, p, b) => {
  const r = await fetch(DRV + p, { method: m, headers: { "Content-Type": "application/json" }, body: b ? JSON.stringify(b) : undefined });
  const j = await r.json(); if (j.value && j.value.error) throw new Error(`${p}: ${j.value.message}`); return j.value;
};
const sid = (await rq("POST", "/session", { capabilities: { alwaysMatch: { browserName: "firefox", "moz:firefoxOptions": { args: ["-headless", "--width=1920", "--height=1300"], binary: "/snap/firefox/current/usr/lib/firefox/firefox" } } } })).sessionId;
const exec = (x) => rq("POST", `/session/${sid}/execute/sync`, { script: x, args: [] });
try {
  await rq("POST", `/session/${sid}/url`, { url: "file://" + resolve(here, "deck.html") });
  await exec(`return document.fonts ? document.fonts.ready.then(() => 1) : 1;`);
  await new Promise((r) => setTimeout(r, 800));
  const slides = await exec(`return document.querySelectorAll(".slide").length;`);
  // 1280x720 CSS px == 33.867 x 19.05 cm at 96 dpi, so each slide fills one page exactly.
  const pdf = await rq("POST", `/session/${sid}/print`, { page: { width: 33.867, height: 19.05 }, margin: { top: 0, bottom: 0, left: 0, right: 0 }, background: true, shrinkToFit: false });
  writeFileSync(resolve(here, "..", "deck.pdf"), Buffer.from(pdf, "base64"));
  // Cover: scale slide 1 by 1.5 inside a 1920x1080 frame, then screenshot that element.
  await exec(`const s = document.querySelector(".slide"); const f = document.createElement("div");
    f.id = "cover-frame"; f.style.cssText = "position:fixed;left:0;top:0;width:1920px;height:1080px;overflow:hidden;z-index:9;background:#07100f";
    const c = s.cloneNode(true); c.style.cssText = "transform:scale(1.5);transform-origin:0 0;";
    f.append(c); document.body.append(f); window.scrollTo(0, 0); return 1;`);
  await new Promise((r) => setTimeout(r, 400));
  const el = await rq("POST", `/session/${sid}/element`, { using: "css selector", value: "#cover-frame" });
  const eid = Object.values(el)[0];
  writeFileSync(resolve(here, "..", "cover.png"), Buffer.from(await rq("GET", `/session/${sid}/element/${eid}/screenshot`), "base64"));
  console.log(`slides=${slides} pdf=${Buffer.from(pdf, "base64").length}B`);
} finally { await rq("DELETE", `/session/${sid}`).catch(() => {}); }
