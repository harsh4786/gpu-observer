// Qwen3-14B's real per-layer kernel sequence, as data -- not its own
// standalone graph anymore (that card was removed; see trace.css/trace.js
// history). The causal graph's GPU EXECUTION node now surfaces this directly
// via currentStageLabel, compressed into one node instead of a whole
// separate section.
//
// This is architecture, not a live CUPTI trace: per-kernel GPU timing is a
// post-run microscope feature (see docs/ARCHITECTURE.md's Join B boundary --
// only reshape_and_cache_flash_kernel has a validated block-to-request
// mapping today; FlashAttention and the dense GEMMs do not). What IS live and
// real is which step is executing and its phase (prefill/decode) and actual
// wall-clock span -- so the sweep through these 11 stages is tagged
// "illustrative", never "measured": a real step is in flight for a real
// duration, but which of the 11 stages is highlighted at any instant is a
// paced animation, not a per-kernel timestamp.
//
// Model constants read from vllm/model_executor/models/qwen3.py:153-170 and
// qwen2.py:110-113 against Qwen/Qwen3-14B's config.json (this session).

export const QWEN3_14B = {
  layers: 40,
  hiddenSize: 5120,
  intermediateSize: 17408,
  numHeads: 40,
  numKvHeads: 8,
  headDim: 128,
  ropeTheta: 1_000_000,
};

// One transformer layer's kernel sequence, in execution order. `attn` carries
// two labels because prefill and decode dispatch genuinely different kernels
// for the same math (FlashAttention varlen vs. paged/flash-decoding) -- see
// this session's engine-step walkthrough.
function layerStages(model) {
  return [
    { title: "input_layernorm", lines: [`RMSNorm · ${model.hiddenSize}`] },
    { title: "qkv_proj", lines: [`${model.hiddenSize} → 7168 · GQA ${model.numHeads}:${model.numKvHeads}`] },
    { title: "q_norm / k_norm", lines: [`RMSNorm · head_dim ${model.headDim}`] },
    { title: "rotary_emb", lines: [`RoPE · θ=${model.ropeTheta.toLocaleString()}`] },
    { title: "reshape_and_cache", lines: ["KV cache write (gpu-observer anchor kernel)"] },
    {
      title: "attn",
      phaseLines: {
        prefill: ["FlashAttention · compute-bound"],
        decode: ["Paged attention · memory-bound"],
      },
      lines: ["attention"],
    },
    { title: "o_proj", lines: [`${model.hiddenSize} → ${model.hiddenSize}`] },
    { title: "post_attention_layernorm", lines: [`RMSNorm · ${model.hiddenSize}`] },
    { title: "gate_up_proj", lines: [`${model.hiddenSize} → ${model.intermediateSize * 2} · SwiGLU`] },
    { title: "SiluAndMul", lines: [`${model.intermediateSize * 2} → ${model.intermediateSize}`] },
    { title: "down_proj", lines: [`${model.intermediateSize} → ${model.hiddenSize}`] },
  ];
}

export const KERNEL_STAGE_COUNT = layerStages(QWEN3_14B).length;

// The 11 stage titles in execution order -- single source of truth for
// anything that needs to label a stage index (e.g. ui/cupti-activity.js's
// real-launch classifier), instead of duplicating this list a second time.
export const STAGE_TITLES = layerStages(QWEN3_14B).map((stage) => stage.title);

// Short labels for the compact stage-chip stepper (causal-graph.js) -- full
// titles don't fit in an 18px chip; the native <title> tooltip carries the
// full name on hover.
export const STAGE_ABBREVIATIONS = [
  "ln", "qkv", "qn", "rope", "kv", "attn", "o", "ln2", "gu", "silu", "dn",
];
