// Records the live session in the browser, so the query YOU just asked can be
// rewound -- no node script, no capture committed to the repo, no canned prompt.
//
// Why this is possible at all: the page already receives every frame it would
// need. The semantic ring (ws 8089) sends one message per engine step, request
// slice and accepted output token; CUPTI (ws 8090) sends one message per real
// kernel launch; vLLM's chat stream sends the decoded text. The live view
// consumes each frame and throws it away. Keeping them costs memory and nothing
// else, and what comes back out is the same measured data the live view drew.
//
// Why keeping them matters: a kernel stage lasts ~0.24ms, a layer ~2.7ms and a
// whole engine step ~106ms, while the browser repaints every 150ms. Live, the
// page can only ever show one arbitrary sample of the ~680 launches since the
// last frame. Replaying what was kept is the only way to step back through the
// sweep without inventing it.
//
// Rewind granularity is one accepted output token, because that is what an
// engine step produces: stepping back is a backspace on the text AND on the
// kernels that produced it, in reverse order, because the seek re-feeds the
// recording up to exactly that token.

import { buildTimeline, createPlayer } from "./replay.js?v=v2g1";

// CUPTI alone runs ~4,500 frames/s on this box, so an unbounded recording of a
// long generation would grow without limit. The cap stops recording and SAYS
// so rather than thinning the stream: dropping frames would quietly break the
// launch counts this project presents as measured.
const DEFAULT_FRAME_LIMIT = 300_000; // ~65s of generation

export function createSessionRecorder({ limit = DEFAULT_FRAME_LIMIT } = {}) {
  const recorder = {
    frames: [],
    recording: false,
    truncated: false,
    prompt: null,
    t0: 0,
  };

  recorder.begin = (prompt) => {
    recorder.frames = [];
    recorder.recording = true;
    recorder.truncated = false;
    recorder.prompt = prompt;
    recorder.t0 = performance.now();
  };

  // Called from the live socket handlers, on the frame they already parsed.
  // The frame object is stored by reference: the live consumers only read it.
  recorder.capture = (src, msg) => {
    if (!recorder.recording) return;
    if (recorder.frames.length >= limit) {
      recorder.recording = false;
      recorder.truncated = true;
      return;
    }
    recorder.frames.push({ t: performance.now() - recorder.t0, src, msg });
  };

  recorder.stop = () => { recorder.recording = false; };
  recorder.count = () => recorder.frames.length;

  // Builds a transport over what was captured. `handlers` routes each frame
  // back to the same consumer the live socket fed, which is what makes the
  // replay the real view code driven by real data, only slower.
  recorder.buildPlayer = (handlers) => {
    const timeline = buildTimeline(recorder.frames);
    const player = createPlayer({
      timeline,
      manifest: {
        prompt: recorder.prompt,
        host: "this session",
        counts: countBySource(recorder.frames),
        truncated: recorder.truncated,
      },
      deliver: (entry) => {
        if (entry.src === "semantic") handlers.semantic?.(entry.msg);
        else if (entry.src === "cupti") handlers.cupti?.(entry.msg);
        else if (entry.src === "chat") {
          const piece = entry.msg?.delta?.choices?.[0]?.delta?.content;
          if (piece) handlers.chat?.(piece);
        }
      },
    });
    player.live = true; // this recording came from the page, not from a file
    return player;
  };

  return recorder;
}

function countBySource(frames) {
  const counts = { semantic: 0, cupti: 0, chat: 0 };
  for (const frame of frames) counts[frame.src] = (counts[frame.src] ?? 0) + 1;
  return counts;
}
