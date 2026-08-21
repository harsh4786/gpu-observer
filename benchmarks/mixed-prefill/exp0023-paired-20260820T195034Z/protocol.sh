#!/usr/bin/env bash
set -euo pipefail

# Five fresh-server, randomized-order paired repetitions of the EXP-0012
# ShareGPT pathology. Performance arms use semantic tracing only; no CUPTI,
# Nsight, sanitizer, or device probes are present.

root=/home/harsh4786/gpu-observer
image=nvcr.io/nvidia/vllm:26.05-py3
model=Qwen/Qwen3-14B
revision=40c069824f4251a91eefaf281ebe4c544efd3e18
interactive="$root/benchmarks/mixed-prefill/exp0012-20260811T143000Z/controlled-workload/interactive-payloads-corrected.jsonl"
background="$root/benchmarks/mixed-prefill/exp0012-20260811T143000Z/controlled-workload/background-long-payloads.jsonl"
aiperf=/home/harsh4786/.venvs/gpu-observer-aiperf-0.10.0/bin/aiperf
semantic_lib="$root/target/release/libgpu_observer_vllm_bridge.so"
semantic_capture="$root/target/release/semantic_capture"
semantic_stats="$root/target/release/semantic_step_stats"
semantic_dump="$root/target/release/semantic_dump"
semantic_core="$root/vllm-adapter/overlay/vllm/v1/engine/core.py"
semantic_py="$root/vllm-adapter/gpu_observer_semantic.py"
packing_runner="$root/vllm-adapter/packed-overlay/vllm/v1/worker/gpu_model_runner.py"
port=8000
pair_count=5
# Randomized once from seed 20260820, then frozen before measurement.
arm_order=(B A A B B A A B B A)
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$root/benchmarks/mixed-prefill/exp0023-paired-$stamp"

mkdir -p "$run_dir/arms"
chmod 0777 "$run_dir" "$run_dir/arms"
cp "$0" "$run_dir/protocol.sh"

current_container=
current_logs_pid=
current_monitor_pid=
current_aiperf_pid=
background_jobs=()
cleanup_arm() {
  if [[ -n "$current_aiperf_pid" ]] && kill -0 "$current_aiperf_pid" 2>/dev/null; then
    kill "$current_aiperf_pid" 2>/dev/null || true
    wait "$current_aiperf_pid" 2>/dev/null || true
  fi
  for job in "${background_jobs[@]}"; do
    kill "$job" 2>/dev/null || true
  done
  background_jobs=()
  if [[ -n "$current_container" ]] && docker ps --format '{{.Names}}' | grep -Fxq "$current_container"; then
    docker stop -t 30 "$current_container" >/dev/null 2>&1 || true
  fi
  if [[ -n "$current_logs_pid" ]] && kill -0 "$current_logs_pid" 2>/dev/null; then
    wait "$current_logs_pid" 2>/dev/null || true
  fi
  if [[ -n "$current_monitor_pid" ]] && kill -0 "$current_monitor_pid" 2>/dev/null; then
    kill "$current_monitor_pid" 2>/dev/null || true
    wait "$current_monitor_pid" 2>/dev/null || true
  fi
  if [[ -n "$current_container" ]]; then
    docker rm "$current_container" >/dev/null 2>&1 || true
  fi
  current_container=
  current_logs_pid=
  current_monitor_pid=
  current_aiperf_pid=
}
trap cleanup_arm EXIT INT TERM

for required in "$interactive" "$background" "$aiperf" "$semantic_lib" \
  "$semantic_capture" "$semantic_stats" "$semantic_dump" "$semantic_core" \
  "$semantic_py" "$packing_runner"; do
  [[ -f "$required" ]] || { echo "missing required artifact: $required" >&2; exit 1; }
done
docker image inspect "$image" >/dev/null

{
  echo "experiment=EXP-0023-paired-mixed-prefill-pathology"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "question=Does frozen long-prefill interference reproducibly degrade interactive tail latency beyond fresh-server variation?"
  echo "design=paired_fresh_server_randomized_arm_order"
  echo "pair_count=$pair_count"
  echo "randomization_seed=20260820"
  echo "frozen_arm_order=${arm_order[*]}"
  echo "arm_A=interactive_only"
  echo "arm_B=interactive_plus_45_fixed_long_prefills"
  echo "performance_instrumentation=semantic_only"
  echo "mechanism_instrumentation=separate_EXP0012_EXP0021_EXP0022_CUPTI_runs"
  echo "model=$model"
  echo "revision=$revision"
  echo "image=$image"
  echo "execution_mode=FULL_AND_PIECEWISE_CUDA_GRAPHS"
  echo "async_scheduling=production_default"
  echo "prefix_caching=false"
  echo "warmup_requests=8"
  echo "measured_interactive_requests=100"
  echo "interactive_concurrency=8"
  echo "interactive_output_tokens=128"
  echo "background_requests=45"
  echo "background_arrival_period_seconds=4"
  echo "background_first_arrival_after_profile_start_seconds=0.5"
  echo "background_output_tokens=16"
  echo "thermal_abort_celsius=85"
  echo "primary_metrics=paired_p99_TTFT_p99_ITL_output_throughput"
  echo "confidence_interval=two_sided_paired_t_95_percent_run_level"
  uname -a
  free -b
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,power.draw --format=csv,noheader
  docker image inspect "$image" --format 'image_id={{.Id}} image_digests={{json .RepoDigests}}'
  "$aiperf" --version
  sha256sum "$interactive" "$background" "$semantic_lib" "$semantic_core" \
    "$semantic_py" "$packing_runner"
} > "$run_dir/manifest.txt"

run_background() {
  local arm_dir=$1
  sleep 0.5
  date +%s%N > "$arm_dir/background/first-injection-realtime-ns.txt"
  for index in $(seq 1 45); do
    payload_line=$(( (index - 1) % 4 + 1 ))
    sed -n "${payload_line}p" "$background" > "$arm_dir/background/request-$index.json"
    (
      curl -fsS --max-time 900 \
        -H 'Content-Type: application/json' \
        -H "X-Request-Id: exp0023-${index}" \
        --data-binary @"$arm_dir/background/request-$index.json" \
        "http://127.0.0.1:$port/v1/chat/completions" \
        -o "$arm_dir/background/response-$index.sse" \
        2> "$arm_dir/background/request-$index.stderr"
    ) &
    background_jobs+=("$!")
    if [[ "$index" -lt 45 ]]; then
      sleep 4
    fi
  done
}

run_arm() {
  local pair=$1
  local arm=$2
  local ordinal=$3
  local arm_dir="$run_dir/arms/pair-$pair-$arm"
  current_container="go-exp0023-p${pair}-${arm,,}-$stamp"
  mkdir -p "$arm_dir/semantic-shm" "$arm_dir/aiperf" "$arm_dir/background"
  chmod 0777 "$arm_dir" "$arm_dir/semantic-shm" "$arm_dir/aiperf" "$arm_dir/background"
  printf 'pair=%s\narm=%s\nordinal=%s\n' "$pair" "$arm" "$ordinal" > "$arm_dir/arm.env"

  docker run -d \
    --name "$current_container" \
    --gpus all \
    --ipc=host \
    --network=host \
    --security-opt label=disable \
    -e GPU_OBSERVER_SEMANTIC_LIB=/observer/libgpu_observer_vllm_bridge.so \
    -e GPU_OBSERVER_SEMANTIC_SHM=/observer-shm/semantic.ring \
    -e GPU_OBSERVER_SEMANTIC_CAPACITY=65536 \
    -e GPU_OBSERVER_MAX_SLICES=1024 \
    -v /home/harsh4786/.cache/huggingface:/root/.cache/huggingface \
    -v "$semantic_lib:/observer/libgpu_observer_vllm_bridge.so:ro" \
    -v "$semantic_core:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py:ro" \
    -v "$semantic_py:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/gpu_observer_semantic.py:ro" \
    -v "$packing_runner:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py:ro" \
    -v "$arm_dir/semantic-shm:/observer-shm" \
    "$image" \
    vllm serve "$model" \
    --revision "$revision" \
    --port "$port" \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --kv-cache-memory-bytes 8G \
    --max-num-seqs 32 \
    --no-enable-prefix-caching > "$arm_dir/container-id.txt"

  docker logs -f "$current_container" > "$arm_dir/server.log" 2>&1 &
  current_logs_pid=$!
  ready=0
  for _ in $(seq 1 900); do
    if curl -fsS -o /dev/null "http://127.0.0.1:$port/health" 2>/dev/null; then
      ready=1
      break
    fi
    if ! docker ps --format '{{.Names}}' | grep -Fxq "$current_container"; then
      break
    fi
    sleep 1
  done
  [[ "$ready" == 1 ]] || { echo "server failed before ready: pair=$pair arm=$arm" >&2; return 1; }
  docker exec "$current_container" chmod 0666 /observer-shm/semantic.ring
  rg -F "Asynchronous scheduling is enabled" "$arm_dir/server.log" > "$arm_dir/async-evidence.txt"
  rg -F "FULL_AND_PIECEWISE" "$arm_dir/server.log" > "$arm_dir/graph-evidence.txt"
  docker top "$current_container" -eo pid,comm,args > "$arm_dir/process-topology.txt"
  engine_host_pid=$(awk '$2 ~ /^VLLM::EngineCor/ {print $1; exit}' "$arm_dir/process-topology.txt")
  [[ "$engine_host_pid" =~ ^[0-9]+$ ]] || return 1

  (
    echo "realtime_ns,mem_available_kb,engine_cpu_pct,engine_rss_kb,engine_threads,gpu_util_pct,gpu_temp_c,gpu_power_w"
    while docker ps --format '{{.Names}}' | grep -Fxq "$current_container"; do
      realtime_ns=$(date +%s%N)
      mem_available_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
      read -r cpu rss threads < <(ps -p "$engine_host_pid" -o %cpu=,rss=,nlwp= | awk 'NF == 3 {print $1,$2,$3}')
      gpu=$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits | head -1 | tr -d ' ')
      echo "$realtime_ns,$mem_available_kb,${cpu:-NA},${rss:-NA},${threads:-NA},$gpu"
      temp=$(cut -d, -f2 <<<"$gpu")
      if [[ "$temp" =~ ^[0-9]+$ ]] && (( temp >= 85 )); then
        : > "$arm_dir/thermal-abort"
        docker stop -t 5 "$current_container" >/dev/null 2>&1 || true
        break
      fi
      sleep 1
    done
  ) > "$arm_dir/system.csv" 2> "$arm_dir/system.stderr" &
  current_monitor_pid=$!

  aiperf_command=(
    "$aiperf" profile
    --model "$model"
    --url "http://127.0.0.1:$port"
    --endpoint-type chat
    --streaming
    --input-file "$interactive"
    --custom-dataset-type raw-payload
    --dataset-sampling-strategy sequential
    --warmup-request-count 8
    --request-count 100
    --concurrency 8
    --artifact-dir "$arm_dir/aiperf"
  )
  printf '%q ' "${aiperf_command[@]}" > "$arm_dir/aiperf-command.txt"
  printf '\n' >> "$arm_dir/aiperf-command.txt"
  "${aiperf_command[@]}" > "$arm_dir/aiperf.log" 2>&1 &
  current_aiperf_pid=$!

  profile_started=0
  for _ in $(seq 1 900); do
    if rg -Fq "Phase profiling started" "$arm_dir/aiperf.log"; then
      profile_started=1
      break
    fi
    kill -0 "$current_aiperf_pid" 2>/dev/null || break
    sleep 0.1
  done
  [[ "$profile_started" == 1 ]] || { echo "AIPerf never entered profiling" >&2; return 1; }
  date +%s%N > "$arm_dir/profile-start-detected-realtime-ns.txt"
  if [[ "$arm" == B ]]; then
    run_background "$arm_dir"
  fi

  set +e
  wait "$current_aiperf_pid"
  aiperf_status=$?
  set -e
  current_aiperf_pid=
  echo "$aiperf_status" > "$arm_dir/aiperf-exit-status.txt"
  [[ "$aiperf_status" == 0 ]] || return 1

  background_status=0
  for job in "${background_jobs[@]}"; do
    wait "$job" || background_status=1
  done
  background_jobs=()
  echo "$background_status" > "$arm_dir/background-exit-status.txt"
  [[ "$background_status" == 0 ]] || return 1
  if [[ "$arm" == B ]]; then
    done_count=$(rg -l '^data: \[DONE\]$' "$arm_dir"/background/response-*.sse | wc -l)
    [[ "$done_count" == 45 ]] || { echo "only $done_count/45 background requests completed" >&2; return 1; }
  fi

  kill "$current_monitor_pid" 2>/dev/null || true
  wait "$current_monitor_pid" 2>/dev/null || true
  current_monitor_pid=
  [[ ! -e "$arm_dir/thermal-abort" ]] || return 1

  "$semantic_capture" "$arm_dir/semantic-shm/semantic.ring" \
    "$arm_dir/semantic.bin" 1 > "$arm_dir/semantic-capture.log" 2>&1
  "$semantic_stats" "$arm_dir/semantic.bin" > "$arm_dir/semantic-step-stats.tsv"
  "$semantic_dump" "$arm_dir/semantic.bin" > "$arm_dir/semantic-dump.log"
  head -1 "$arm_dir/semantic-step-stats.tsv" \
    | rg 'sequence_gaps=0 loss_markers=0' > "$arm_dir/semantic-quality.txt"
  if rg -i 'GPU observer semantic emitter disabled|EngineCore encountered a fatal|CUDA error|misaligned address' "$arm_dir/server.log"; then
    return 1
  fi

  report=$(find "$arm_dir/aiperf" -type f -name profile_export_aiperf.json -print -quit)
  [[ -n "$report" ]] || return 1
  cp "$report" "$arm_dir/metric-summary.json"
  jq -e '.request_count.avg == 100 and (.error_summary | length) == 0 and
    .output_sequence_length.count == 100 and .output_sequence_length.min == 128 and
    .output_sequence_length.max == 128 and .total_output_tokens.avg == 12800' \
    "$arm_dir/metric-summary.json" > "$arm_dir/aiperf-integrity.json"

  docker logs "$current_container" > "$arm_dir/server-final.log" 2>&1 || true
  docker stop -t 30 "$current_container" >/dev/null
  wait "$current_logs_pid" 2>/dev/null || true
  current_logs_pid=
  docker rm "$current_container" >/dev/null
  current_container=
  chmod -R a+rX "$arm_dir" 2>/dev/null || true
  echo "pair=$pair arm=$arm status=pass" | tee "$arm_dir/outcome.txt"
}

for ordinal in "${!arm_order[@]}"; do
  pair=$((ordinal / 2 + 1))
  arm=${arm_order[$ordinal]}
  run_arm "$pair" "$arm" "$((ordinal + 1))"
done

printf 'pair\tarm\tordinal\tttft_p99_ms\titl_p99_ms\tthroughput_tok_s\tttft_avg_ms\titl_avg_ms\tduration_s\tmixed_steps\n' \
  > "$run_dir/run-metrics.tsv"
for pair in $(seq 1 "$pair_count"); do
  for arm in A B; do
    arm_dir="$run_dir/arms/pair-$pair-$arm"
    ordinal=$(awk -F= '$1 == "ordinal" {print $2}' "$arm_dir/arm.env")
    mixed_steps=$(awk -F '\t' '$1 == "mixed" {print $2}' "$arm_dir/semantic-step-stats.tsv")
    jq -r --arg pair "$pair" --arg arm "$arm" --arg ordinal "$ordinal" \
      --arg mixed_steps "$mixed_steps" \
      '[$pair,$arm,$ordinal,.time_to_first_token.p99,.inter_token_latency.p99,
        .output_token_throughput.avg,.time_to_first_token.avg,
        .inter_token_latency.avg,.benchmark_duration.avg,$mixed_steps] | @tsv' \
      "$arm_dir/metric-summary.json" >> "$run_dir/run-metrics.tsv"
  done
done

echo "result=pass" > "$run_dir/outcome.env"
echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$run_dir/outcome.env"
"$root/benchmarks/seal-run.sh" "$run_dir"
trap - EXIT INT TERM
printf '%s\n' "$run_dir"
