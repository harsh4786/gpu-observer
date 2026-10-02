// Replays a capture recorded by ui-v2/tools/record-session.mjs, at any speed.
//
// Why this exists: on this box a kernel stage lasts ~0.24ms, a layer ~2.7ms and
// a whole engine step ~106ms, while the browser repaints every 150ms. Nothing
// inside a step is observable live -- the page can only ever sample one of the
// ~680 launches that happened since the last frame. Slowing a recording down is
// the only way to show the sweep without inventing it: every frame played here
// is a real captured event with its own timestamp.
//
// It works by standing in for the transport, not by changing the code that
// consumes it. Before connectWebSocket() / connectCuptiActivity() run, the
// global WebSocket is replaced with a stand-in that delivers recorded frames,
// and fetch is wrapped so the chat request yields the recorded token stream.
// Neither trace.js's patch handling nor cupti-activity.js's classifier can tell
// the difference, which is the point: what you watch is the real view code
// driven by real data, only slower.
//
// Timing comes from each frame's OWN clock, not from when it arrived:
//   cupti     msg.startNs   device-side nanoseconds
//   semantic  msg.ts        host monotonic nanoseconds
//   chat      frame.t       arrival ms (the stream has no clock of its own)
// Arrival order is bursty -- CUPTI flushes every 200ms, so hundreds of launches
// land at once -- and replaying by arrival would stretch those bursts instead of
// the sweep. Each source is normalised against its own first frame, because the
// two clocks have different epochs and cannot be compared directly.

const PORT_SEMANTIC = "8089";
const PORT_CUPTI = "8090";

async function fetchSession(url, gzipped) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`replay session: HTTP ${response.status} for ${url}`);
  if (!gzipped) return response.text();
  if (typeof DecompressionStream !== "function") throw new Error("no DecompressionStream");
  const stream = response.body.pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).text();
}

function nsOf(value) {
  // startNs runs past 2^53, so the subtraction has to happen in BigInt or the
  // differences come back quantised.
  try {
    return BigInt(value);
  } catch {
    return null;
  }
}

function buildTimeline(frames) {
  const bySource = { semantic: [], cupti: [], chat: [] };
  for (const frame of frames) {
    const list = bySource[frame.src];
    if (list) list.push(frame);
  }
  const timeline = [];
  for (const [src, list] of Object.entries(bySource)) {
    if (!list.length) continue;
    const clockKey = src === "cupti" ? "startNs" : src === "semantic" ? "ts" : null;
    let origin = null;
    if (clockKey) {
      for (const frame of list) {
        const value = nsOf(frame.msg?.[clockKey]);
        if (value !== null) { origin = value; break; }
      }
    }
    for (const frame of list) {
      let at = frame.t; // fallback: arrival order
      if (clockKey && origin !== null) {
        const value = nsOf(frame.msg?.[clockKey]);
        if (value !== null) at = Number(value - origin) / 1e6;
      }
      timeline.push({ at, src, msg: frame.msg });
    }
  }
  timeline.sort((a, b) => a.at - b.at);
  return timeline;
}

class ReplaySocket {
  constructor(url) {
    this.url = String(url);
    this.readyState = 1;
    this.listeners = new Map();
    queueMicrotask(() => this.emit("open", {}));
  }
  addEventListener(type, handler) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(handler);
  }
  removeEventListener(type, handler) {
    const list = this.listeners.get(type);
    if (list) this.listeners.set(type, list.filter((entry) => entry !== handler));
  }
  emit(type, event) {
    for (const handler of this.listeners.get(type) ?? []) handler(event);
  }
  deliver(msg) {
    if (this.readyState === 1) this.emit("message", { data: JSON.stringify(msg) });
  }
  close() {
    if (this.readyState === 3) return;
    this.readyState = 3;
    this.emit("close", {});
  }
}

export async function installReplay(url) {
  const base = new URL(url, location.href);
  const sessionUrl = base.href.endsWith(".ndjson")
    ? base.href
    : new URL("session.ndjson", base.href.replace(/\/?$/, "/")).href;
  const manifestUrl = new URL("manifest.json", sessionUrl.replace(/[^/]+$/, "")).href;

  const [sessionText, manifest] = await Promise.all([
    // A capture is ~2MB per second raw and about 6% of that gzipped, so the
    // committed copy is compressed and the raw file is only a fallback for
    // captures taken straight off the recorder. DecompressionStream does the
    // work in the browser, which keeps this a static file on any plain host --
    // no server-side content negotiation to arrange.
    fetchSession(`${sessionUrl}.gz`, true).catch(() => fetchSession(sessionUrl, false)),
    fetch(manifestUrl).then((response) => (response.ok ? response.json() : null)).catch(() => null),
  ]);

  const frames = [];
  for (const line of sessionText.split("\n")) {
    if (!line.trim()) continue;
    try { frames.push(JSON.parse(line)); } catch { /* a truncated tail is not fatal */ }
  }
  const timeline = buildTimeline(frames);

  const sockets = { semantic: null, cupti: null };
  const RealWebSocket = window.WebSocket;
  window.WebSocket = function ReplayWebSocket(target) {
    const socket = new ReplaySocket(target);
    const text = String(target);
    if (text.includes(PORT_SEMANTIC)) sockets.semantic = socket;
    else if (text.includes(PORT_CUPTI)) sockets.cupti = socket;
    // Anything else (the shadow sanitizer ports) gets a socket that simply
    // never delivers, which is what a capture without those sources means.
    return socket;
  };
  window.WebSocket.OPEN = RealWebSocket?.OPEN ?? 1;

  const player = {
    manifest,
    frameCount: timeline.length,
    speed: 1,
    playing: false,
    virtualMs: 0,
    cursor: 0,
    chatWaiters: [],
    chatQueue: [],
    chatDone: false,
    onProgress: null,
  };

  const pumpChat = () => {
    while (player.chatWaiters.length && (player.chatQueue.length || player.chatDone)) {
      const resolve = player.chatWaiters.shift();
      if (player.chatQueue.length) resolve({ done: false, value: player.chatQueue.shift() });
      else resolve({ done: true, value: undefined });
    }
  };

  const encoder = new TextEncoder();
  const deliver = (entry) => {
    if (entry.src === "semantic") sockets.semantic?.deliver(entry.msg);
    else if (entry.src === "cupti") sockets.cupti?.deliver(entry.msg);
    else if (entry.src === "chat") {
      if (entry.msg.kind === "done") {
        player.chatDone = true;
        player.chatQueue.push(encoder.encode("data: [DONE]\n\n"));
      } else if (entry.msg.kind === "delta") {
        player.chatQueue.push(encoder.encode(`data: ${JSON.stringify(entry.msg.delta)}\n\n`));
        // Also handed over directly: the fetch shim only runs when something
        // submitted the chat form, but pressing play has to show the text too.
        const piece = entry.msg.delta?.choices?.[0]?.delta?.content;
        if (piece) player.onChatDelta?.(piece);
      }
      pumpChat();
    }
  };

  let lastTick = 0;
  const tick = (now) => {
    if (!player.playing) return;
    const delta = lastTick ? now - lastTick : 0;
    lastTick = now;
    player.virtualMs += delta * player.speed;
    while (player.cursor < timeline.length && timeline[player.cursor].at <= player.virtualMs) {
      deliver(timeline[player.cursor]);
      player.cursor += 1;
    }
    player.onProgress?.(player.cursor, timeline.length);
    if (player.cursor >= timeline.length) {
      player.playing = false;
      player.chatDone = true;
      pumpChat();
      return;
    }
    requestAnimationFrame(tick);
  };

  // One accepted output token per engine step, so these are the natural rewind
  // stops: stepping back one is a backspace on the decoded text, and because
  // the seek re-feeds the capture up to that point, the kernels, layer sweep
  // and stage counts land exactly where they stood when that token was emitted.
  const tokenMarks = [];
  timeline.forEach((entry, index) => {
    if (entry.src === "semantic" && entry.msg?.kind === "accepted_output_token") tokenMarks.push(index);
  });
  player.tokenCount = tokenMarks.length;
  player.tokensEmitted = () => tokenMarks.filter((index) => index < player.cursor).length;

  // Rewinding cannot un-apply events -- every consumer accumulates -- so a seek
  // resets them all and replays from the start at no delay. 15k frames costs a
  // few tens of ms, which is cheap enough to do on a button press.
  // Stepping always leaves the recording paused. Resuming after a seek made the
  // rewind invisible: it stepped back one token and then immediately raced
  // forward again from there, which reads as a restart from the beginning.
  player.seekToToken = (count) => {
    const target = Math.max(0, Math.min(tokenMarks.length, count));
    const stop = target === 0 ? 0 : tokenMarks[target - 1] + 1;
    player.playing = false;
    player.onReset?.();
    player.cursor = 0;
    player.virtualMs = 0;
    player.chatQueue = [];
    player.chatDone = false;
    for (let index = 0; index < stop; index += 1) deliver(timeline[index]);
    player.cursor = stop;
    player.virtualMs = stop > 0 ? timeline[stop - 1].at : 0;
    player.onProgress?.(player.cursor, timeline.length);
  };

  player.start = () => {
    if (player.playing) return;
    player.playing = true;
    lastTick = 0;
    requestAnimationFrame(tick);
  };
  player.pause = () => { player.playing = false; };
  player.setSpeed = (value) => { player.speed = value; };
  player.restart = () => {
    player.cursor = 0;
    player.virtualMs = 0;
    player.chatQueue = [];
    player.chatDone = false;
  };

  const realFetch = window.fetch.bind(window);
  window.fetch = (input, init) => {
    const target = typeof input === "string" ? input : input?.url ?? "";
    if (!target.includes("/v1/chat/completions")) return realFetch(input, init);
    // The page "sends" a message; what comes back is the recorded stream. The
    // prompt typed is ignored -- the marker says which prompt was captured.
    player.restart();
    player.start();
    return Promise.resolve({
      ok: true,
      status: 200,
      body: {
        getReader: () => ({
          read: () => new Promise((resolve) => {
            player.chatWaiters.push(resolve);
            pumpChat();
          }),
          releaseLock() {},
          cancel() { player.pause(); },
        }),
      },
    });
  };

  return player;
}

// A replay must never be mistaken for a live session: the data is measured, the
// timing is reproduced, and at 0.02x what you are watching took 50x less time
// than it appears to.
export function showReplayMarker(player) {
  const manifest = player.manifest;
  const bar = document.createElement("div");
  bar.id = "replay-marker";
  bar.className = "replay-marker";
  const when = manifest?.recordedAt
    ? new Date(manifest.recordedAt).toISOString().replace("T", " ").slice(0, 19)
    : "unknown time";
  const host = manifest?.host ?? "unknown host";
  const counts = manifest?.counts
    ? `${manifest.counts.cupti ?? 0} kernel launches · ${manifest.counts.semantic ?? 0} engine frames`
    : `${player.frameCount} frames`;
  const prompt = manifest?.prompt;
  bar.innerHTML = `<strong>REPLAY</strong> recorded ${when} UTC on ${host} · ${counts}`
    + (prompt ? ` · prompt: <em>${prompt.replace(/[<&]/g, (c) => (c === "<" ? "&lt;" : "&amp;"))}</em>` : "")
    + ` · <span id="replay-speed">1x</span>`;
  document.querySelector(".app")?.prepend(bar);
  return bar;
}
