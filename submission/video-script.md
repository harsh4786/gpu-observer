# Video script — GPU Observer (target 4:30, hard limit 5:00)

Record at 1920×1080 on your Mac (QuickTime → File → New Screen Recording),
browser full-screen, voice-over live. Export as MP4 under 300 MB.

**Before recording:** ask Claude to relaunch the live stack, open the 4-port
tunnel, and hard-refresh `http://127.0.0.1:9088/ui/trace.html`. Pause any gpu_ext
experiments: they stop the demo containers.

---

## 0:00–0:25 · The problem  (screen: README top, or deck slide 2)

> When you send a prompt to an LLM server, it disappears into a black box. The
> scheduler batches it with other requests, the GPU runs packed and reordered
> rows, and a profiler shows you kernels — but not whose work they are. You
> can't answer a simple question: what did *my* request actually do on the GPU?

## 0:25–0:50 · What this is  (screen: deck slide 3, the chain diagram)

> GPU Observer keeps a request's identity all the way down: prompt, tokens,
> scheduler step, the GPU rows it was packed into, the real CUDA kernels that
> ran, and the KV-cache blocks it owns. It runs on an NVIDIA DGX Spark with vLLM
> and Qwen3-14B — and every number is labeled by how it was measured.

## 0:50–2:30 · Live demo  (screen: live UI over the tunnel)

1. **0:50** Point at the chat box. Type: *"Explain in two sentences what a KV cache is."* Send.
   > The prompt goes straight into the graph.
2. **1:00** The **prompt** arrow lights up; the **tokens** node fills in.
   > That's the real tokenized length, read from the scheduler.
3. **1:10** The **step** and **rows** nodes appear.
   > Each engine step — prefill first, then decode — and the exact GPU rows this request occupied.
4. **1:25** Scroll to the **kernel DAG**; nodes pulse.
   > These are the 11 kernel stages of one transformer layer, lit by real CUPTI kernel launches — not an animation. It repeats 40 times per token.
5. **1:50** The **kernel calls this query** bars.
   > Every kernel call this request has made, split into thinking tokens and response tokens.
6. **2:10** Scroll to **decoded output**; point at "tokens GPU-confirmed".
   > And the text you see is cross-checked against tokens the GPU actually accepted.

## 2:30–3:25 · Sealed trace  (screen: https://harsh4786.github.io/gpu-observer/)

1. **2:30** Open the hosted URL. Click **Load sample trace**.
   > This is a real capture you can open yourself — no GPU needed.
2. **2:40** Change the **Graph step** selector to a step with several lanes.
   > Here our request shares the GPU with seven others. Where the colored edges cross, vLLM reordered rows between the scheduler and the GPU — which is exactly why naive attribution goes wrong.
3. **3:00** Click the **GPU node**; show **Kernel work ownership**.
   > And this is the payoff: this kernel's real GPU interval, and the cache blocks that belong to our request — reconstructed with the row-to-block rule we validated against actual device events.
4. **3:15** Point at **What each edge means**.
   > Every claim is labeled measured, reconstructed, or unavailable.

## 3:25–4:10 · Results  (screen: deck results slide)

> On the KV-cache kernel under CUDA Graphs, we recovered per-request block
> ownership across 7,560 device events with zero mismatches, and joined the same
> requests to real GPU kernel intervals with 64-nanosecond calibration. And it's
> cheap: across 15 fresh-server runs, no instrumentation setting — including
> capturing every kernel launch — was distinguishable from plain vLLM within half
> a percent.

## 4:10–4:30 · Limits and close  (screen: deck final slide)

> Exact ownership is proven for one kernel family today, and the live kernel view
> is tuned to Qwen3-14B. Next is generalizing it to any model. Try the trace
> yourself at harsh4786.github.io/gpu-observer.
