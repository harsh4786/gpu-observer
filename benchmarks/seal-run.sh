#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: benchmarks/seal-run.sh RUN_DIRECTORY" >&2
  exit 2
fi

benchmark_root=$(cd "$(dirname "$0")" && pwd -P)
run_dir=$(realpath -e "$1")

case "$run_dir" in
  "$benchmark_root"/*) ;;
  *)
    echo "refusing to seal a directory outside $benchmark_root" >&2
    exit 2
    ;;
esac

temporary=$(mktemp "$run_dir/.checksums.XXXXXX")
(
  cd "$run_dir"
  find . -type f \
    ! -name checksums.sha256 \
    ! -name '.checksums.*' \
    -print0 |
    sort -z |
    xargs -0 -r sha256sum
) > "$temporary"

mv "$temporary" "$run_dir/checksums.sha256"
sync -f "$run_dir/checksums.sha256"
echo "sealed $run_dir"
