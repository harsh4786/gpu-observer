# Design iterations: the live causal-graph UI

**For**: codex, or whoever picks up `ui/` work next. See `HANDOVER.md` at the
repo root first — it explicitly says design/UI work should be handed to a
spun-up Claude Code session, not done cold. This file is the context that
session (or you, if you're reading it directly) needs to not re-make the
same mistakes already corrected once.

**Files involved**: `ui/trace.html`, `ui/trace.css`, `ui/trace.js`,
`ui/causal-graph.js`, `ui/graph-primitives.js`, `ui/kernel-activity.js`,
`ui/cupti-activity.js`, `ui/kernel-graph.js`.

## The arc, in one sentence

The graph went from a single vague "GPU EXECUTION" box → an elaborate
11-node kernel-by-kernel snake diagram (rejected as cluttered) → a
deliberately restrained hero-panel-plus-ticker design → two new real-data
visuals layered on top of that restrained base (a 40-cell per-layer sweep
rail, and a 48-cell SM-occupancy grid). Every step was driven by direct,
often sharp user feedback, not by an internal design review — read the
feedback quotes below literally, they're the actual bar this UI has to
clear.

## Round 1 — first complaints about the original graph

User feedback (verbatim): *"the graph is too cluttered... squint my eye...
decode step reached too quickly... no detailed activity on the gpu kernel
size"*. This was about the pre-existing lane-based causal graph (query →
tokenizer → enginecore → gpumodelrunner → GPU execution → outcome), which
worked fine for the sealed offline trace but read as noisy once live-mode
was added on top of it.

Then, a substantial structural ask: *"putting more emphasis on... foreground
query / tokenizer... engine core, focused rows and focused slice and gpu
model runner (those two nodes are important but not as much as the active
control flow)... expand the gpu execution node out into a neural network
type visualization with the nodes as kernels and edges as the flow of
execution."* Follow-up, terser: *"make it graph like."*

**What was built**: an 11-node node-link diagram in `causal-graph.js`
(`kernelNodePosition`, `TOP_ROW`/`BOTTOM_ROW`, a snake layout — 6 nodes top
row left-to-right, 5 bottom row right-to-left, explicit directional edges
between consecutive kernel stages, a labeled "×40 layers" loop-back edge).
Self-caught bug before it even reached the user: the first version of the
bottom-row column math didn't align with the top row, so the loop-back edge
would have swooped diagonally across the whole diagram — fixed by deriving
`KERNEL_BOTTOM_GAP` so the 5-node bottom row spans the exact same total
width as the 6-node top row.

## Round 2 — dead/redundant UI trimmed

Three separate, specific complaints, each acted on directly:

- *"remove the decode bar on top and 'accepted output'/'sass unavailable'
  nodes... examine why and remove if not necessary"* — removed after
  confirming they were dead weight, not load-bearing for any real data path.
- *"whats the 'no launch observed yet' subtext... why is it there? if we
  dont need it we can remove it"* — this was overlapping other text and
  also removed.
- *"each node looks way cluttered wiht text"* — this complaint recurred
  **twice** at different points (see Round 8's tooltip lesson below); the
  first time, it meant literally: stop putting `title` + up to 2 `lines` +
  a `badge` on every one of the 11 kernel nodes. Fixed by stripping kernel
  nodes down to `title` only, no `lines`, no `badge`.

## Round 3 — output card as a downstream visual

User idea (verbatim, mid-message): *"make the textbox visual as downstream
to the graphs output as the network decodes each token, its a great
inutitibe design."* Implemented: `trace.html` restructured so `.chat-card`
holds only the input form, and a new `#output-connector` + `.output-card`
(holding the reply text) sits immediately below `.graph-card`, visually
framing decoded text as flowing out of the graph rather than living in an
unrelated sibling card. `trace.js` toggles `.output-connector.active` at
`step_begin`/`step_end` so the connector itself shows live state, not just
a static line.

## Round 4 — "have design taste"

*"how to access claude design?, the graph is shit... looks cheap and
distasteful."* Also, in the same exchange: *"the sizing here is inverted,
the top one should look bigger and the smaller arrow diagram, you have to
take care of things like these and have some design taste."* And: *"make
the title bigger."*

This is the point where the user explicitly pushed back on being asked too
many clarifying questions and wanted decisive craft calls instead. Response
was to stop asking and just decide: bumped `h1` size/weight, fixed the
inverted size hierarchy (the thing that should read as primary was smaller
than a secondary element), added scoped `.topbar .eyebrow` sizing,
adjusted `.app`/`.topbar` spacing. **Lesson for whoever picks this up**:
when this user says "have design taste," they mean stop presenting options
and make the call — the cost of a wrong call is a quick correction, the
cost of another question is friction they've already flagged twice.

## Round 5 — the caching bug (most consequential bug of the whole project)

*"somethings seriously wrong with the design here. its not gone and also
the previous runs decode steps still there, so refresh doesnt do anything."*

**Root cause**: only the top-level `<script src="trace.js?v=graphN">` tag
in `trace.html` was version-bumped on each change. The internal ES module
imports (`import ... from "./causal-graph.js"`, etc.) had no query string at
all, so the browser served a stale cached copy of every file *except*
`trace.js` itself — no matter how many times the top-level tag's version
was bumped. This had been silently defeating **every UI fix reported as
shipped for multiple rounds before this one** — the user was correctly
reporting broken behavior, and every one of those "fixes" really had never
reached the browser.

**Fix, and the standing rule since**: every internal `import ...
"./x.js?v=graphN"` across every `ui/*.js` file must carry the exact same
version string as the top-level tag, bumped together, every single time any
`ui/*.js` file changes. Current files touched by this rule:
`trace.html` (link + script tag), `trace.js`, `causal-graph.js`,
`cupti-activity.js`, `kernel-activity.js` — verify with
`grep -rn "v=graphN" ui/` before calling any UI change done. Current
version as of this writing: `graph24`.

A second, related bug found in the same round: `ensureLiveTrace()` only
reset live-trace state on the *first* call per page load (`if
(state.liveMode) return;`), so every subsequent chat message in the same
browser session appended onto the same never-cleared trace instead of
starting fresh. Fixed by removing the guard — it has exactly one caller
(`sendChatMessage`) that should always want a clean start.

## Round 6 — skills, audit, and the big simplification

The user asked to install three third-party Claude Code skill packages
(`anthropics/skills` frontend-design, `nexu-io/open-design`,
`vercel-labs/agent-skills` web-design-guidelines), then: *"yes, do an audit
of the design using those skills, and then make it simplistic while keeping
the semantics of the design."*

**This is where the 11-node snake diagram from Round 1 was deliberately
replaced.** The audit concluded the elaborate node-link graph, however
correctly it represented the real kernel flow, was doing too much visual
work for a live-glance UI. It was replaced with the current, much more
restrained pattern still in place today:

- A compact upstream request-flow strip (Query / Tokens / Scheduler /
  Packed as small single-line boxes) — de-emphasized on purpose, real
  context but secondary to the actual point of the view.
- A single "NOW EXECUTING" hero panel — one large card showing the most
  recent real kernel launch's title, detail line (kernel name + grid dims),
  and a `layer N/40` badge.
- A short horizontal history ticker (`addStageChip`, up to 14 entries) of
  the last several distinct stage transitions, each a small abbreviated
  chip (`STAGE_ABBREVIATIONS` — `ln`, `qkv`, `qn`, `rope`, `kv`, `attn`,
  `o`, `ln2`, `gu`, `silu`, `dn`) with a native `<title>` tooltip carrying
  the full name.

**Lesson**: this user's taste converges on *fewer, bigger, real-data-driven
elements* over *many small elements that are individually accurate*. The
11-node graph was architecturally more complete but read as worse UI. Don't
assume "shows more real data" beats "reads cleanly at a glance" — ask which
one this specific user wants before building the elaborate version, or at
least be ready to simplify fast.

A specific implementation lesson from this round, still relevant: the
`.stage-chip`/`.graph-node` idle-state stroke was changed to
`color-mix(in srgb, var(--node-color) 55%, transparent)` — a calmer,
lower-contrast default state so idle elements don't visually compete with
the active/live ones. This "idle should visually recede, live should pop"
principle recurred later (see Round 8).

## Round 7 (this session) — two new real-data visuals added

New user direction: *"the graph is nowhere near where we want it to be...
we can represent those 40 layers in some graph visual if we can see
activations per layer and then we can also represent these kernels on the
hardware itself by visualizing sm grids on the gpu."*

Two important technical-grounding steps happened **before** any code was
written, both worth repeating as a pattern:

1. **Verified the real SM count** rather than guessing — `torch.cuda.
   get_device_properties(0).multi_processor_count` inside the project's own
   CUDA image confirmed **48 SMs** for this GB10. The grid is sized to that
   real number, not a round guess.
2. **Verified the sanitizer's actual capability** before designing around
   an assumption — `GPU_OBSERVER_SAN_KERNEL_SUBSTRING` is a single
   fixed-size `strstr` match in `subscriber.cpp` (confirmed by reading the
   C++, not assumed), meaning the SM grid can only ever be "real" for
   whichever *one* kernel the shadow container currently has patched, never
   several simultaneously. This constraint is stated directly in the UI's
   subtitle text so it's never silently oversold as more complete than it
   is.

**Layer sweep rail** (`causal-graph.js`): 40 small cells below the hero/
ticker, one per transformer layer, lit once that layer's first real launch
lands this step, with the currently-executing layer pulsing
(`.stage-chip.active`). Data comes from a new retention layer added to
`cupti-activity.js` — it previously only kept the *most recent* launch per
stage (overwriting each layer's data every launch), so `layerLog` (a real
40-entry array, reset on every `step_begin`) had to be added specifically
for this. Layer boundary is detected honestly: every layer's first kernel
is always classified as `INPUT_LAYERNORM` (either `rsqrt_2` for layers
1-39, or the fused `embedding_rms_norm` for layer 0), so incrementing a
counter on that classification is a real signal, not inferred.

**Visual-language decision worth flagging**: the default `.stage-chip`
styling (inherited from the ticker) is stroke-only — deliberately subtle,
per Round 6's "idle should recede" principle. That's fine for 14 sequential
ticker chips with a native tooltip each, but was **too faint to read as a
40-cell heatmap at a glance**. Rather than change the shared primitive
(which would regress the ticker), a scoped `.layer-sweep .stage-chip.
measured rect { fill: ... }` rule was added, wrapping just the 40 layer
cells in their own `<g class="layer-sweep">` group. **Lesson**: a visual
treatment that's correct for one component (a sequential ticker) is not
automatically correct for a structurally different one (an at-a-glance
heatmap/rail), even when they share a primitive — scope the override
instead of either regressing the original or duplicating the primitive.

**SM occupancy grid** (`kernel-activity.js` + `trace.html`/`trace.css`): a
new 48-cell grid inside the existing `kernel-activity-card`, one cell per
SM index, fed by the shadow container's real `sm_id` field. First version:
class computed fresh every 250ms from a decay-window calculation, full
`innerHTML` rebuild of all 48 cells every tick.

## Round 8 — "sm occupancy grid is dead" (three real findings in one bug report)

This single complaint led to three separate real findings, not one fix —
worth reading in full since it's a good example of how "not working" can
have layered causes that each need independent verification, not one guess.

**Finding 1, real but misread as a bug**: sniffed the raw WebSocket traffic
directly (`python3` raw-socket WS client — no headless browser was
available on this box; see "no browser tooling" note below) and confirmed
`reshape_and_cache_flash_kernel` genuinely launches with grid `(1,1,1)` —
exactly one block per launch. During decode, only 1 of 48 SMs is ever
active at a literal instant, and the GPU's own block scheduler often reuses
the same SM for long stretches. **This is real hardware behavior, not a
bug** — during prefill (multi-block launches), the same kernel showed real
SM diversity (block indices and `sm_id` spread across 0-15 in one capture).

**Finding 2, the actual code bug**: the decay design faded a "touched"
cell's styling all the way back to being visually identical to a
never-touched cell within a couple seconds. Combined with Finding 1's
genuinely-sparse real occupancy, the grid looked permanently empty even
though real data had landed. **Fix, and a reusable principle**: separate
"has this ever fired" (a permanent low-tint `.touched` class that never
clears) from "did this fire *recently*" (a temporal glow layered on top).
Collapsing both into one decaying state is what made real, low-frequency
data look broken — anywhere a real signal is naturally sparse, don't let
"idle" and "never happened" collapse into the same visual state.

**Finding 3, not a code bug at all**: after Finding 2's fix still didn't
resolve it, direct network inspection (`ss -tnp` on the box itself) showed
**zero established connections** to the shadow container's WebSocket ports
(8189/8190), while the primary container's ports (8089/8090) had live
connections. The user's SSH tunnel simply wasn't forwarding the newer
shadow-container ports yet — a networking gap on the user's side, not a UI
bug. **Lesson**: when a live-data UI element looks dead, check the
transport layer (is a connection even established?) before re-diagnosing
the rendering code a third time. `ss -tnp | grep :<port>` from the same box
answers this in one command and would have saved a full round-trip earlier
if checked first.

**Aside — no browser tooling available on this box**: `claude-in-chrome`
isn't connected here, and neither `puppeteer` nor `playwright` (Node or
Python) are installed. Every visual verification this session had to be
either (a) inferred from server-side signals (WS traffic sniffed with a
throwaway raw-socket Python client, port connection state via `ss`, log
greps) or (b) confirmed by the user directly describing what they saw. If a
future session has real browser access, use it — it would have shortened
several of these rounds significantly.

## Round 9 — "activate those SMs per launch like a blink"

Direct ask, plus an invitation to brainstorm additional real metadata.
Implementation change: the grid moved from "rebuild all 48 divs from
scratch every 250ms, computing a decay-based class" to **persistent DOM
cells built once**, updated per-event with a real CSS-animation restart
trick:

```js
cell.classList.remove("blink");
void cell.offsetWidth;   // force a reflow so the browser "forgets" the animation ran
cell.classList.add("blink");
```

This matters specifically because the anchor kernel often hits the *same*
SM on consecutive launches (see Round 8, Finding 1) — without the forced
reflow, a repeat hit on an already-`blink`-classed element wouldn't restart
the CSS animation, so only the *first* hit on any given SM would ever
visibly flash.

Folded into the same change, since the data was already computed and just
unused: each blink is colored by the **real owning request** (reusing
`requestColor()`, which was moved out of `causal-graph.js` into the shared
`graph-primitives.js` so both modules draw from one consistent palette
instead of inventing a second one), and tagged with the **real layer
index** the event belongs to (free to derive: this kernel fires exactly
once per layer, so the existing per-step event counter *is* the layer
index, no new tracking needed).

Brainstormed but not yet built (still real, still grounded in fields
already flowing over the wire, no new instrumentation needed): grouping
same-timestamp events to distinguish a decode-phase launch (1 block) from a
prefill-phase launch (many blocks, sharing one `ts`); a "hottest SM" badge;
a live launches/second cadence stat; promoting the existing
`sequence_errors`/`drops` quality counters from a hover-only tooltip to an
always-visible reliability readout.

## Round 10 — "the blocks dont show the sm numbers, and they should in a lighter way"

Smallest round, but specific: added the SM index as always-visible text
inside each cell (`cell.textContent = String(sm)`), styled deliberately
dim at rest (`color: var(--dim); font: 7px ui-monospace`) so it doesn't
compete with the color-coded fill, brightening slightly once `.touched`,
and flipping to dark-on-bright (`var(--bg)`) during the `.blink` flash
itself for contrast against the lit background.

## Current final visual architecture (as of `?v=graph24`)

Top to bottom in `.graph-card`'s SVG (`causal-graph.js`'s `renderLiveGraph`):

1. Compact upstream strip: Query → Tokens → Scheduler slice → Packed rows.
2. "NOW EXECUTING" hero panel: title, kernel name + grid dims detail line,
   `layer N/40` badge, `mint` = measured / `cyan` = illustrative-fallback
   color coding.
3. History ticker: up to 14 abbreviated stage chips, oldest to newest.
4. **Layer sweep rail**: 40 cells, real per-layer completeness this step,
   current layer pulsing.

Separately, in `.kernel-activity-card` (fed by the shadow container):

5. Metric row (layers this step / events this step / events this session),
   a progress bar, and a scrolling recent-events feed (kernel name,
   attributed request, SM id, block index).
6. **SM occupancy grid**: 48 cells (real GB10 SM count), each showing its
   index, blinking on every real block-entry event, colored by owning
   request, permanently tinted once touched, hover tooltip carrying layer
   index + event count + last owner.

## Reusable principles, extracted (read this before making the next change)

- **Version every internal ES module import together, every time.** This
  is not optional — see Round 5. `grep -rn "v=graphN" ui/` before calling
  anything done.
- **This user's taste is fewer/bigger/real over many/small/complete.** When
  in doubt, simplify rather than add — see Round 6.
- **Idle state should visually recede; live state should pop.** Applies at
  both the component level (Round 6's stroke-opacity change) and the
  per-datum level (Round 8's touched/recent split).
- **A shared visual primitive is not automatically correct for a
  structurally different use** (sequential ticker vs. at-a-glance heatmap)
  — scope an override rather than regress the original or fork the
  primitive (Round 7).
- **When real data is naturally sparse, don't let "quiet" collapse into
  looking identical to "never happened."** Separate a permanent
  evidence-of-occurrence state from a temporal recency glow (Round 8).
- **Check the transport layer before re-diagnosing rendering code a third
  time.** `ss -tnp | grep :<port>` on the box answers "is anything even
  connected" in one command (Round 8, Finding 3).
- **Never fabricate GPU facts to fill a gap** — this predates the UI work
  (see `HANDOVER.md`'s two hard architectural facts) but shows up directly
  in the UI: the SM grid's subtitle explicitly states it's scoped to one
  kernel type, not silently implying full coverage.
- **No headless browser is available on this box.** Verify live-data UI
  changes via server-side signals (WS sniff, `ss`, log greps) or by asking
  the user precisely what they see — don't guess at DOM/rendering behavior
  you can't actually observe.
