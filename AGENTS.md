# Engineering invariants

These instructions apply to this repository.

- Keep `observer-core` and `policy-engine` `no_std`.
- Use `alloc` only for explicit bounded or fallible startup/cold-path storage.
- Do not add `String`, `Vec`, trait objects, reference-counted pointers, or
  OS handles to `EventRecord`.
- Do not add locks or unbounded channels to probe emission.
- One producer owns one SPSC ring. Cross-source merging belongs in the collector.
- A full telemetry path drops visibly; it never blocks inference.
- Variable-length relationships use flat records and dense index ranges.
- Preserve raw timestamps, source clocks, sequences, and raw events.
- Never substitute host launch duration for GPU execution duration.
- Report summed kernel time and overlap-safe busy time separately.
- Any hot-path optimization needs a benchmark; any new allocation needs a
  measured memory budget.
- Run `cargo test --workspace` and the release ring benchmark after changing
  record layout, transport, or correlation.
- Keep hardware-specific C/C++ boundaries narrow and outside
  `observer-core`.
