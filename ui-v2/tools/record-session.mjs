#!/usr/bin/env node
// Records one live capture session to disk so it can be replayed later, and
// replayed SLOWLY.
//
// The reason this exists is a measurement, not convenience. On this box the GPU
// runs ~4,500 kernel launches a second: a kernel stage lasts ~0.24ms, a layer
// ~2.7ms, a whole engine step ~106ms. The browser repaints every 150ms. So at
// real speed nothing inside a step is observable -- the live view can only ever
// show one arbitrary sample out of the ~680 launches that happened since the
// last frame. Recording every frame with its own timestamp is the only way the
// sweep can be shown honestly: played back at 1/100 speed, each of the 440
// launches in a step is a real captured event, not an interpolation.
//
// Captures the two live sources the UI consumes:
//   semantic ring  ws://127.0.0.1:8089  engine steps, request and packed slices,
//                                       accepted output tokens
//   CUPTI activity ws://127.0.0.1:8090  one frame per real kernel launch
// and, unless --no-chat, the vLLM chat stream it drives, so the decoded text
// replays too.
//
// Both sources carry their own high-resolution clock (CUPTI startNs/endNs,
// semantic ts/seq). Those are preserved untouched inside each frame; the `t`
// this writer adds is only arrival order, for sources that have no clock of
// their own. A player should prefer the embedded timestamps.
//
//   node ui-v2/tools/record-session.mjs --seconds 4 --prompt "Count to twenty."
//
// Size: CUPTI alone is ~4,500 frames/s. Keep captures short -- a single step is
// ~106ms, so 2-4 seconds is already dozens of steps. The run prints the byte
// count so it can be checked before anything is committed.

import { createWriteStream, mkdirSync, statSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const WebSocket = require("ws"); // node 18 has no global WebSocket client

const HELP = `record-session.mjs -- capture one live session for replay

  --out DIR         output directory (default ui-v2/data/replay-<stamp>)
  --seconds N       how long to record after the sockets open (default 4)
  --prompt TEXT     prompt to send (default "Count to twenty slowly.")
  --max-tokens N    cap on generated tokens (default 64)
  --no-chat         record the sockets only; drive traffic yourself
  --semantic URL    default ws://127.0.0.1:8089
  --cupti URL       default ws://127.0.0.1:8090
  --vllm URL        default http://127.0.0.1:8000
  --model NAME      default Qwen/Qwen3-14B`;

function parseArgs(argv) {
  const args = {
    out: null,
    seconds: 4,
    prompt: "Count to twenty slowly.",
    maxTokens: 64,
    chat: true,
    semantic: "ws://127.0.0.1:8089",
    cupti: "ws://127.0.0.1:8090",
    vllm: "http://127.0.0.1:8000",
    model: "Qwen/Qwen3-14B",
  };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const next = () => argv[(i += 1)];
    if (flag === "--out") args.out = next();
    else if (flag === "--seconds") args.seconds = Number(next());
    else if (flag === "--prompt") args.prompt = next();
    else if (flag === "--max-tokens") args.maxTokens = Number(next());
    else if (flag === "--no-chat") args.chat = false;
    else if (flag === "--semantic") args.semantic = next();
    else if (flag === "--cupti") args.cupti = next();
    else if (flag === "--vllm") args.vllm = next();
    else if (flag === "--model") args.model = next();
    else if (flag === "--help" || flag === "-h") { console.log(HELP); process.exit(0); }
    else { console.error(`unknown flag: ${flag}`); process.exit(2); }
  }
  return args;
}

function connect(name, url, onFrame) {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(url);
    socket.on("open", () => resolve(socket));
    socket.on("error", (error) => reject(new Error(`${name} (${url}): ${error.message}`)));
    socket.on("message", (data) => {
      // Kept as the parsed object rather than the raw string so the file is one
      // well-formed JSON object per line. A frame that does not parse is still
      // recorded, flagged, rather than dropped -- a silently thinned capture
      // would break counts this project presents as measured.
      const text = data.toString();
      try {
        onFrame(name, JSON.parse(text));
      } catch {
        onFrame(name, { kind: "__unparsed", raw: text });
      }
    });
  });
}

async function recordChat(args, onFrame) {
  const response = await fetch(`${args.vllm}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: args.model,
      messages: [{ role: "user", content: args.prompt }],
      max_tokens: args.maxTokens,
      stream: true,
    }),
  });
  if (!response.ok || !response.body) throw new Error(`chat HTTP ${response.status}`);
  const decoder = new TextDecoder();
  let buffer = "";
  for await (const chunk of response.body) {
    buffer += decoder.decode(chunk, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, cut).trim();
      buffer = buffer.slice(cut + 1);
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (payload === "[DONE]") { onFrame("chat", { kind: "done" }); return; }
      try {
        onFrame("chat", { kind: "delta", delta: JSON.parse(payload) });
      } catch {
        onFrame("chat", { kind: "__unparsed", raw: payload });
      }
    }
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
  const outDir = path.resolve(args.out ?? `ui-v2/data/replay-${stamp}`);
  mkdirSync(outDir, { recursive: true });
  const ndjsonPath = path.join(outDir, "session.ndjson");
  const stream = createWriteStream(ndjsonPath);

  const counts = { semantic: 0, cupti: 0, chat: 0 };
  let started = 0;
  const onFrame = (src, msg) => {
    counts[src] = (counts[src] ?? 0) + 1;
    stream.write(`${JSON.stringify({ t: +(performance.now() - started).toFixed(3), src, msg })}\n`);
  };

  console.log(`  recording to ${outDir}`);
  const sockets = await Promise.all([
    connect("semantic", args.semantic, onFrame),
    connect("cupti", args.cupti, onFrame),
  ]);
  started = performance.now();
  console.log(`  sockets open; capturing ${args.seconds}s`);

  const chatDone = args.chat
    ? recordChat(args, onFrame).catch((error) => console.error(`  chat failed: ${error.message}`))
    : Promise.resolve();

  await new Promise((resolve) => setTimeout(resolve, args.seconds * 1000));
  await chatDone;
  for (const socket of sockets) { try { socket.close(); } catch { /* already closed */ } }
  await new Promise((resolve) => stream.end(resolve));

  const durationMs = +(performance.now() - started).toFixed(1);
  const bytes = statSync(ndjsonPath).size;
  const manifest = {
    version: 1,
    recordedAt: new Date().toISOString(),
    host: os.hostname(), // HOSTNAME is not exported in every shell
    model: args.model,
    prompt: args.chat ? args.prompt : null,
    durationMs,
    counts,
    bytes,
    sources: {
      semantic: args.semantic,
      cupti: args.cupti,
      chat: args.chat ? `${args.vllm}/v1/chat/completions` : null,
    },
    // A player must know which clock to trust; `t` is arrival order only.
    clocks: {
      t: "ms since capture start, arrival order, added by the recorder",
      cupti: "msg.startNs / msg.endNs -- CUPTI device-side nanoseconds",
      semantic: "msg.ts -- nanoseconds on the host monotonic clock; msg.seq is the ring sequence",
    },
  };
  writeFileSync(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);

  const rate = (counts.cupti / (durationMs / 1000)).toFixed(0);
  console.log(`  captured ${durationMs}ms`);
  console.log(`    semantic ${counts.semantic}  cupti ${counts.cupti} (${rate}/s)  chat ${counts.chat}`);
  console.log(`    ${(bytes / 1e6).toFixed(2)} MB -> ${ndjsonPath}`);
  if (counts.cupti === 0) console.log("  WARNING: no CUPTI frames -- is the stream server up on 8090?");
  if (counts.semantic === 0) console.log("  WARNING: no semantic frames -- is the stream server up on 8089?");
}

main().catch((error) => { console.error(error); process.exit(1); });
