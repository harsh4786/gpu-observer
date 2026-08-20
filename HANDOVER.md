# Handover: gpu-observer

**Written by**: Claude Code session, 2026-08-20
**For**: codex, picking this project back up
**Repo**: https://github.com/harsh4786/gpu-observer (private)

## Read this first: design work routing

**If the next task involves the UI (`ui/*.js`, `ui/trace.html`, `ui/trace.css`,
the live causal graph, the SM-occupancy grid, or any visual/design decision),
do not do that work yourself. Spin up a Claude Code session in this
repo and hand the task to it instead.**

Why: this session did the entire live-UI build (the per-layer sweep rail, the
SM-occupancy grid with per-launch blink and request-color attribution, the
hero/ticker redesign) and has the design-skill tooling and the full context of
*why* each visual decision was made — including several rounds of the user
rejecting cluttered/over-designed attempts before converging on the current
restrained style. Reimplementing that context from scratch in a different
agent risks regressing the same clutter mistakes already corrected once. Point
Claude at this file plus `ui/` and let it continue from here.

Everything else — Rust/C++ (`collector/`, `cupti-agent/`,
`device-probes/`, `host-probes/`, `vllm-adapter/`), experiment validation,
kernel-safety gating, infra/deploy scripts — is exactly the kind of work this
project has been doing with codex; keep doing that part yourself.

## Project north star

Programmable request-to-GPU observability for vLLM V1, built and measured on
a single NVIDIA DGX Spark (GB10 GPU, **48 SMs**, compute capability 12.1,
121GB **unified** CPU/GPU memory — not discrete VRAM, so a `gpu-memory-
utilization` cap is required per replica whenever more than one vLLM instance
runs concurrently, or the default computation against total device memory
will refuse to start / OOM). Model under test throughout: Qwen3-14B, BF16,
`revision=40c069824f4251a91eefaf281ebe4c544efd3e18`.

The core question the project answers, end to end and only from real
measured evidence (never fabricated): for a given chat request, which GPU
kernels actually ran, on which streaming multiprocessors, for how long, and
were they that request's own work or someone else's (packed-batch
attribution). Every claim in this codebase is tagged with its real evidence
tier (`measured` / `reconstructed` / `matched` / `illustrative` /
`unavailable`) — never silently upgraded.

## Two hard architectural facts everything else follows from

1. **CUPTI and NVIDIA Compute Sanitizer cannot share one process** on this
   stack (`CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED`, confirmed
   EXP-0020). CUPTI gives real per-launch grid/block/stream/duration for
   *every* kernel type; the sanitizer gives real per-block SM occupancy but
   only for whichever *one* kernel substring it's patched with
   (`GPU_OBSERVER_SAN_KERNEL_SUBSTRING` is a single fixed-size `strstr`
   match in `device-probes/compute-sanitizer/subscriber.cpp:99,840` — not a
   list). This is why the live demo runs as **two separate vLLM
   containers** mirroring every request, not one.
2. **Compute Sanitizer instrumentation under CUDA Graph replay is
   crash-prone and config-specific**, not something you can assume works for
   a new kernel. EXP-0016 found two independent callback-identity modes both
   producing a CUDA misaligned-address fault in production (async
   scheduling + FULL_AND_PIECEWISE graphs). EXP-0018 found the fix: stable
   *function-scoped* callback userdata (`GPU_OBSERVER_SAN_CALLBACK_DATA_
   SCOPE=function`) plus separate graph-node identity tracking
   (`GPU_OBSERVER_SAN_GRAPH_NODES=1`) — and explicitly **not**
   `GPU_OBSERVER_SAN_LAUNCH_ID`, which is the exact config that faulted.
   That combination is the only validated-safe sanitizer config in this
   repo. Every kernel validation this session reused it verbatim.

## Live two-container demo — how to run it

```bash
cd /home/harsh4786/gpu-observer
bash benchmarks/run-live-demo-cupti.sh    # primary: CUPTI, all 11 kernel stages, :8000/:8089/:8090
bash benchmarks/run-live-demo-shadow.sh   # shadow: sanitizer, one anchor kernel, :8001/:8189/:8190
# serve the UI (or use the persistent gpu-observer-ui container, already running on :8088)
python3 -m http.server 8088 --directory /home/harsh4786/gpu-observer
# open http://127.0.0.1:8088/ui/trace.html?v=graph24  (bump ?v= if you edit ui/*.js — see below)
```

**Operational gotcha discovered this session, twice**: both scripts are
long-running (`wait $pid1 $pid2` at the end) with an `EXIT` trap that
`docker stop`/`docker rm`s their container the moment the wrapping bash
process dies. When run via a backgrounded tool call, launch them with the
harness's own backgrounding (`run_in_background: true`) and **no extra
`&`/`nohup` of your own** — stacking your own `&` on top races with the
harness's process reaping and kills the container within seconds, before
its trap can be avoided. Also: a session boundary killed both containers
cleanly twice in a row (each time was a legitimate `EXIT` trap firing, not a
crash) — expect to relaunch both after any harness/session restart, and run
the smoke test again before trusting them.

**Second gotcha, more serious**: at least once, killing the wrapping script
didn't kill the actual `VLLM::EngineCore` process — it got orphaned outside
any container's cgroup and kept holding 79GB of GPU memory for hours,
causing the next launch to fail with an OOM-shaped error
(`Free memory ... is less than desired GPU memory utilization`). Before
relaunching after any failure, check `nvidia-smi --query-compute-apps=
pid,used_memory --format=csv` for a process with no matching `docker ps`
container, and `kill -9` it if found.

**Mandatory smoke test before ever pointing a browser at it** (this
project's own standing rule, not new): one curl chat request to each
container, then grep both `server.log` files for `"misaligned address"` and
`"EngineCore encountered a fatal"` — must find nothing.

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-14B","messages":[{"role":"user","content":"hi"}],"max_tokens":10}' > /dev/null
curl -sS http://127.0.0.1:8001/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-14B","messages":[{"role":"user","content":"hi"}],"max_tokens":10}' > /dev/null
grep -rn "misaligned address\|EngineCore encountered a fatal" \
  benchmarks/live-demo/*-cupti/server.log benchmarks/live-demo/*-shadow/server.log
```

**Cache-busting**: `ui/trace.html`'s top-level `<script>`/`<link>` tags AND
every internal `import ... from "./x.js?v=graphN"` must carry the *same*
version string, bumped together, every time any `ui/*.js` file changes. This
bit us hard earlier in the session — only the top-level tag was versioned
for several rounds, so fixes silently never reached the browser while
looking like they had shipped. Current version: `graph24`.

## What this session built (chronological)

1. **CUPTI kernel-name classifier** (`ui/cupti-activity.js`) — empirically
   captured, not guessed, mapping of Qwen3-14B's 11 per-layer kernel stages
   to real mangled kernel names. Four stages (qkv_proj/o_proj/gate_up_proj/
   down_proj) share the literal same `gemvx::kernel` symbol during decode;
   disambiguated by `grid_x` where unique (qkv_proj=1792, gate_up_proj=8704)
   and otherwise by launch-position heuristics (o_proj follows the attn
   group, down_proj follows SiluAndMul).
2. **Fixed a real collector bug**: `collector/src/join_b_trace.rs`'s
   `MAX_DEVICE_BYTES` was `64 * 1024 * 1024`, one 64-byte header short of a
   completely full sanitizer ring's actual size — silently rejecting valid,
   well-formed capture files. Fixed by doubling the constant; verified via a
   new permanent diagnostic binary, `collector/examples/debug_device_
   events.rs` (takes a capture file path, prints parsed event count and
   quality counters — keep using this to sanity-check any new capture).
3. **UI redesign, several iterations**: started from a cluttered 11-node
   snake-layout kernel graph, went through multiple simplification passes
   (removed always-visible per-node text, removed a broken "no launch
   observed yet" state, fixed a critical stale-state bug in
   `ensureLiveTrace()`), and converged on the current restrained language: a
   compact upstream request strip, a single "now executing" hero panel, a
   short history ticker, and (new this session) two real-data visual
   additions:
   - **Layer sweep rail** (`ui/causal-graph.js`) — 40 cells, one per
     transformer layer, lit from `cuptiSnapshot.layerLog` (a new per-layer
     retention `cupti-activity.js` didn't have before: it used to keep only
     the *most recent* launch per stage, discarding full-step history).
   - **SM occupancy grid** (`ui/kernel-activity.js`) — a 48-cell grid (real
     SM count for this GB10, confirmed via
     `torch.cuda.get_device_properties(0).multi_processor_count`, not
     guessed), each cell showing its SM index, blinking per real block-entry
     event (persistent DOM nodes + a CSS-animation-restart trick so
     back-to-back hits on the same SM still flash, not just the first one),
     colored by the real owning request (reused `requestColor()`, moved into
     the shared `graph-primitives.js` so both modules use one palette), and
     tagged with the real layer index the event belongs to (derived free
     from the existing per-step event counter, since this kernel fires
     exactly once per layer).
   - Along the way, diagnosed the SM grid reading as "dead" twice: once
     because the decay design faded touched cells back to looking identical
     to never-touched ones (fixed with a permanent low-tint "touched" class
     separate from the recency glow), and once because the browser's SSH
     tunnel simply wasn't forwarding the shadow container's ports yet
     (confirmed by directly sniffing the raw WebSocket traffic with a
     throwaway Python client — no headless browser was available on this
     box, `claude-in-chrome` isn't connected here, and `puppeteer`/
     `playwright` aren't installed; a raw-socket WS sniff was the only way
     to verify server-side behavior without user-side DevTools access).
4. **Kernel validation round 2** (see below) — 4 more kernels confirmed
   safe, 2 found to genuinely deadlock (not crash) under CUDA Graph capture.
5. **Repo bootstrapped and pushed to GitHub** — this directory had never
   been a git repo. Initialized, extended `.gitignore` to exclude
   experiment run output (`benchmarks/live-demo/`, `*.activities.tsv`,
   `*.events.bin`, `*.ring`, `.cache/`, the vendored `third_party/aiperf`
   embedded repo, nested `host-probes/ebpf/target/`) — without this,
   several GB of binary experiment artifacts and one 642MB dataset cache
   would have been committed, and at least one generated file (140MB)
   exceeded GitHub's 100MB hard limit. Pushed via `gh` device-code auth
   (the box's only prior SSH key was scoped for a different server, and the
   `gh` token was stale) to `harsh4786/gpu-observer`, private.

## Kernel validation round 2 (this session, not a registered EXP)

Followed the project's existing one-kernel-at-a-time discipline (see the
`gpu-observer-future-sanitizer-kernels` deferred-work item this round
picked up), consolidated to 5 substring experiments since some substrings
cover multiple pipeline stages:

| Substring | Stages covered | Result |
|---|---|---|
| `SiluAndMul` | SiluAndMul | **safe** |
| `rsqrt` | input_layernorm, post_attention_layernorm | **safe** |
| `q_norm`/`k_norm` | QK norm | **safe** |
| `rotary_emb` | rotary position embedding | **safe** |
| `gemvx` | qkv_proj, o_proj, gate_up_proj, down_proj | **NOT usable** — hung (0% GPU util, `futex_do_wait`) immediately after "Autotuning process ends", never reached CUDA graph capture. Killed after 13+ min, no crash signature. |
| `flash_fwd` | attn (FlashAttention) | **NOT usable** — got further: `PIECEWISE` (prefill) graph capture completed cleanly in ~1.5s, then hung identically before/during `FULL` (decode-only) graph capture. Killed after 6+ min, no crash signature. |

**Pattern, not yet root-caused**: both failures are the highest
launch-frequency/most-complex kernel types in the model (the dominant GEMV
projections, and attention itself) and both stall at the exact same
structural boundary — `FULL` graph capture. Every kernel that validated
clean is a simpler, lower-volume elementwise or normalization op. Working
hypothesis: the sanitizer's `block_event` callback path deadlocks
specifically when instrumenting a high-launch-volume kernel during `FULL`
graph capture — plausibly a synchronization conflict between the
per-block event-write callback and CUDA's own graph-capture stream
semantics under high launch density. **Not investigated further** (would
need to go inside the sanitizer's C++ / NVIDIA's own capture internals) —
this is the concrete next step if `gemvx`/`flash_fwd` SM-level coverage is
ever wanted. Full writeup: memory `gpu-observer-future-sanitizer-kernels`.

Net effect: 5 of 11 per-layer stages (attn, qkv_proj, o_proj, gate_up_proj,
down_proj) cannot get live sanitizer (L4, block/SM-level) telemetry with the
current instrumentation approach, and shouldn't be faked — represent them
as "L4 unavailable for this kernel type," matching the project's existing
evidence-tier discipline.

## Full experiment registry (EXP-0001 through EXP-0020)

Source of truth: `benchmarks/registry.toml` — read it directly for anything
below that needs more than the one-line summary; each entry links to a full
`manifest.toml`/`claims.md`/`analysis.md` under its `artifact_path`.

| ID | Title | Result |
|---|---|---|
| EXP-0001 | Cold vs. prefix-cached CUDA launch-path selection | Both paths execute 378 kernels; 84 launches migrate `cuLaunchKernelEx`→`cuLaunchKernel`. |
| EXP-0002 | Live vLLM engine-step semantics joined to external CUDA submissions (async) | Semantic ring stayed empty — NGC 26.05 auto-enabled async scheduling, bypassing `EngineCore.step`; host probe still captured 1,143 launches. |
| EXP-0003 | Same, synchronous engine-step mode | All 1,146 CUDA submissions assigned to 3 semantic steps, zero loss, after explicit PID-namespace normalization. |
| EXP-0004 | CUPTI 13.2.1 clock calibration/injection gate | 3 API + 3 kernel records, matching correlation IDs; 64-sample clock offset spread 16ns. |
| EXP-0005 | Engine-step GPU timing, eager vs. CUDA Graphs | Graph mode compressed 1,146 host launches to 134 launch APIs; GPU kernel time fell 36.1% in the matched trace. |
| EXP-0006 | Qwen3-14B engine-step wall decomposition (CUPTI) | 1,614 device kernels assigned to 3 steps; GPU busy 376.895ms of 381.053ms step wall time. |
| EXP-0007 | bpftime / gpu_ext compatibility on GB10 | bpftime works after PTX 9.0/CUDA-13 fixes; `gpu_ext` live attach blocked — stock NVIDIA 580 driver lacks the needed hooks/BTF types. |
| EXP-0008 | Compute Sanitizer overhead ladder, Qwen3-14B BF16 eager | Block callbacks matched 344,907,690 expected blocks exactly; throughput impact -0.22% to -2.91% across modes. |
| EXP-0009 | Qwen3-14B production CUDA Graph compatibility gate | Subscriber + all-module no-op survived graphs; stateful all-module counters crashed (misaligned address); one allowlisted GEMM counter completed clean. |
| EXP-0010 | AIPerf saturation sweep, clean vs. selected counter | Saturates at concurrency 32 (64 adds 0.36% throughput, +3.39x p99 TTFT); allowlisted counter: -1.09% throughput over 3.2M callbacks. |
| EXP-0011 | BurstGPT P90 5-min replay | 271 requests, arrival timing preserved to 1.6ms, never materially queued; counter: -0.14% throughput, +0.64% avg ITL over 9.47M callbacks. |
| EXP-0012 | *(not in registry.toml — directory only: `benchmarks/mixed-prefill/exp0012-*`)* Interference between long/compute-heavy and short interactive requests sharing an engine step | Central pathology hypothesis of the whole project; demonstrated causally once — check the directory directly, not summarized here. |
| EXP-0013 | Join B packed-row ownership oracle | 58,400 block events had exact coverage, but scheduler-order attribution was rejected: `InputBatch` compaction reordered 8 request rows in one step. |
| EXP-0014 | Join B with authoritative packed-row events | Fixed EXP-0013's gap: 58,400 block events across 760 launches, zero loss/mismatch, across 19 steps including one full 8-row reorder. |
| EXP-0015 | Join B observability overhead ladder | -0.33% median throughput vs. clean, -0.28% incremental SASS cost; 1,466,880 device events across 62,400 launches, zero Join B errors. |
| EXP-0016 | Production CUDA Graph SASS block-event compatibility gate | **Rejected** — both per-launch and function-scoped callback identity faulted (misaligned address) under graph replay. Full writeup: `benchmarks/join-b/EXP0016-REPORT.md`. |
| EXP-0017 | Production packed-row attribution under AIPerf ShareGPT | 100 requests, zero errors under async CUDA Graphs; all 1,805 steps had complete packed layouts; one 4-request step reordered every position. |
| EXP-0018 | Graph-safe SASS block attribution (the fix for EXP-0016) | Stable function-scoped callback userdata eliminated the fault: 114,112 block events, zero drops, 280 complete graph-node records. This is the config every later kernel validation reused. |
| EXP-0019 | End-to-end CUDA Graph request-to-SASS block ownership | 8 concurrent requests, 35 decode replays, 7,560 block events; 6,400 request-owned + 1,160 padding blocks across 25 reordered steps, zero drops/mismatches. |
| EXP-0020 | CUPTI-timed request-to-cache-kernel correlation | 35 decode steps, 1,400 kernel intervals, zero drops/mismatches; confirmed CUPTI+sanitizer cannot run same-run (one subscriber slot) — the fact the two-container architecture is built on. |

## Known open items / next steps

- Root-cause the `gemvx`/`flash_fwd` `FULL`-graph-capture deadlock (see
  above) if SM-level coverage for those 5 stages is ever wanted.
- The live architecture's Part 3 (shadow container request-mirroring — done)
  and Part 4 (request-attribution card — done, this session extended it with
  color + layer tagging) are complete; Part 2's main-graph redesign is done
  in a simplified form (hero+ticker, not the originally-planned 11-node
  snake diagram — that was deliberately replaced after user feedback that it
  was too cluttered). Check `/home/harsh4786/.claude/plans/cheerful-
  greeting-barto.md` if it still exists for the original plan text, but
  trust the actual `ui/` code over it — it's decayed.
- No CI, no tests beyond `cargo test --release --workspace` (Rust side) —
  nothing currently gates the UI's correctness beyond manual smoke-testing
  and `node --input-type=module --check` for syntax.
- The repo is private; nothing about visibility has been discussed beyond
  that default choice.
