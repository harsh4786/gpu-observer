#!/usr/bin/env bash
set -euo pipefail

root=/home/harsh4786/gpu-observer
run_id=$(date -u +%Y%m%dT%H%M%SZ)
experiment_root="$root/benchmarks/join-b/exp0015-overhead-$run_id"
rung="$root/benchmarks/run-join-b-overhead-rung.sh"

[[ -x "$rung" ]] || { echo "missing rung runner: $rung" >&2; exit 1; }
if docker ps --format '{{.Names}}' | grep -q .; then
  echo "refusing to start while another Docker container is running" >&2
  docker ps --format 'name={{.Names}} status={{.Status}}' >&2
  exit 2
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
  echo "refusing to start while another GPU compute process is active" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2 || true
  exit 2
fi

mkdir -p "$experiment_root"
{
  echo "experiment=EXP-0015"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "question=What is the incremental client-visible cost of scheduler semantics, authoritative packed-row semantics, and selected SASS block events?"
  echo "repetitions=3"
  echo "ordering=counterordered"
  echo "arm_clean=no_observer_patch"
  echo "arm_scheduler=engine_step_and_scheduler_slices"
  echo "arm_packed=scheduler_plus_authoritative_packed_rows"
  echo "arm_sass=packed_plus_selected_cache_kernel_block_events"
  echo "arm_cupti=packed_plus_unfiltered_cupti_activity_capture"
  echo "note=cupti and sass are alternatives on top of packed, not successive rungs"
  sha256sum "$rung"
} > "$experiment_root/manifest.txt"

order=(
  $'1\t1\tcupti'
  $'2\t1\tsass'
  $'3\t1\tpacked'
  $'4\t1\tscheduler'
  $'5\t1\tclean'
  $'6\t2\tclean'
  $'7\t2\tscheduler'
  $'8\t2\tsass'
  $'9\t2\tpacked'
  $'10\t2\tcupti'
  $'11\t3\tpacked'
  $'12\t3\tcupti'
  $'13\t3\tscheduler'
  $'14\t3\tclean'
  $'15\t3\tsass'
)
printf 'ordinal\trepeat\tarm\n%s\n' "${order[@]}" > "$experiment_root/run-order.tsv"

while IFS=$'\t' read -r ordinal repeat arm; do
  [[ "$ordinal" == ordinal ]] && continue
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting ordinal=$ordinal repeat=$repeat arm=$arm"
  "$rung" "$experiment_root" "$arm" "$repeat"
done < "$experiment_root/run-order.tsv"

{
  echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader
} > "$experiment_root/final-state.txt"

echo "completed overhead ladder: $experiment_root"
