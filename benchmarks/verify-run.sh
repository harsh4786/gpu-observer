#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: benchmarks/verify-run.sh RUN_DIRECTORY" >&2
  exit 2
fi

benchmark_root=$(cd "$(dirname "$0")" && pwd -P)
run_dir=$(realpath -e "$1")

case "$run_dir" in
  "$benchmark_root"/*) ;;
  *)
    echo "refusing to verify a directory outside $benchmark_root" >&2
    exit 2
    ;;
esac

cd "$run_dir"
sha256sum --check checksums.sha256
