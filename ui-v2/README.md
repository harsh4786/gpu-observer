# ui-v2 — timeline-first redesign

Work in progress. `../ui/` stays the shipped page and is not touched by this
folder; everything here is a copy that the redesign lands in, so the live demo
and the hosted viewer keep working while this is built.

Open `trace.html` the same way as the v1 page:

```bash
python3 -m http.server 8088 --directory .          # from the repo root
# live:   http://127.0.0.1:8088/ui-v2/trace.html
# sealed: http://127.0.0.1:8088/ui-v2/trace.html?offline=1
```

The sealed sample is read from `../ui/data/sample-qwen3-14b-trace-v2.json`
rather than copied, because it is 7.2 MB.

## What is different so far (phase 1)

- **Timeline spine** replaces the step dropdown: one tick per engine step, tick
  height is scheduled tokens on a log scale, so a prefill step stands above a
  run of single-token decodes. Phase uses a neutral ramp; mint, amber, violet
  and red stay reserved for evidence tiers.
- **Transport**: play, pause, step back and forward, speed 1x/2x/4x.
- **Live pin**: the playhead follows the newest step; a manual seek detaches it
  and "jump to live" re-attaches, the log-viewer contract.
- Ticks are appended rather than rebuilt (each step owns a fixed slice of the
  SVG viewBox), so a 400-step live run costs one `<rect>` per step.

Phases 2 to 5 (stage restructure, present mode, recorded replay) are in the
plan and not built yet.
