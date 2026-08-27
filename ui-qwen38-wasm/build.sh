#!/usr/bin/env bash
# Two build modes, matching the explicit "always release, thin LTO for fast
# iteration, fat LTO once we're sure about the design" instruction:
#   ./build.sh          -- release profile, lto=thin, codegen-units=16
#   ./build.sh --fat    -- release-fat profile, lto=fat, codegen-units=1,
#                          plus a wasm-opt -O3 pass. Much slower to build,
#                          smaller/faster-at-runtime output -- only worth it
#                          right before committing to a design, not every
#                          edit.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out_dir="$root/../ui-qwen38/pkg"

profile=release
fat=0
if [[ "${1:-}" == "--fat" ]]; then
  profile=release-fat
  fat=1
fi

cd "$root"
cargo build --target wasm32-unknown-unknown --profile "$profile"

wasm_file="$root/target/wasm32-unknown-unknown/$profile/ui_qwen38_wasm.wasm"
mkdir -p "$out_dir"
wasm-bindgen --target web --out-dir "$out_dir" --out-name ui_qwen38 "$wasm_file"

if [[ "$fat" == "1" ]]; then
  if command -v wasm-opt >/dev/null 2>&1; then
    wasm-opt -O3 "$out_dir/ui_qwen38_bg.wasm" -o "$out_dir/ui_qwen38_bg.wasm"
    echo "wasm-opt -O3 applied"
  else
    echo "wasm-opt not found on PATH -- skipping the extra optimization pass (binaryen's wasm-opt, not installed on this box yet)" >&2
  fi
fi

echo "built ($profile) -> $out_dir"
ls -la "$out_dir"
