#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 EXPERIMENT_ROOT {clean|subscriber|block_noop|block_counter|block_event|sampled_memory_barrier|full_memory} [KERNEL_SUBSTRING]" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage

experiment_root=$(realpath -m "$1")
mode=$2
kernel_substring=${3:-}

case "$mode" in
  clean|subscriber|block_noop|block_counter|block_event|sampled_memory_barrier|full_memory) ;;
  *) usage ;;
esac

root=/home/harsh4786/gpu-observer
image=gpu-observer/vllm-sanitizer:cuda13.2
model=Qwen/Qwen3-14B
revision=40c069824f4251a91eefaf281ebe4c544efd3e18
port=8000
seed=20260809
warmup_prompts=${GO_SAN_WARMUP_PROMPTS:-8}
warmup_output_tokens=${GO_SAN_WARMUP_OUTPUT_TOKENS:-16}
measured_prompts=${GO_SAN_MEASURED_PROMPTS:-64}
measured_output_tokens=${GO_SAN_MEASURED_OUTPUT_TOKENS:-64}
max_concurrency=${GO_SAN_MAX_CONCURRENCY:-8}
event_capacity=${GO_SAN_EVENT_CAPACITY:-65536}
sample_log2=${GO_SAN_SAMPLE_LOG2:-12}
launch_identity=${GO_SAN_LAUNCH_ID:-0}
container="go-san-${mode}-$(date -u +%H%M%S)"
run_dir="$experiment_root/$mode"

if [[ -e "$run_dir" ]]; then
  echo "refusing to overwrite existing rung: $run_dir" >&2
  exit 2
fi
mkdir -p "$run_dir/semantic-shm"
chmod 0777 "$run_dir" "$run_dir/semantic-shm"

semantic_lib="$root/target/release/libgpu_observer_vllm_bridge.so"
semantic_capture="$root/target/release/semantic_capture"
semantic_stats="$root/target/release/semantic_step_stats"
join_b_cache="$root/target/release/join_b_cache"
semantic_core="$root/vllm-adapter/overlay/vllm/v1/engine/core.py"
semantic_py="$root/vllm-adapter/gpu_observer_semantic.py"
sanitizer_build="$root/device-probes/compute-sanitizer/build"

for required in "$semantic_lib" "$semantic_capture" "$semantic_stats" \
                "$semantic_core" "$semantic_py" \
                "$sanitizer_build/libgpu_observer_sanitizer.so" \
                "$sanitizer_build/gpu_observer_sanitizer_patches.cubin"; do
  [[ -f "$required" ]] || { echo "missing required artifact: $required" >&2; exit 1; }
done

monitor_pid=
logs_pid=
cleanup() {
  if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  if docker ps --format '{{.Names}}' | grep -Fxq "$container"; then
    docker stop --time 30 "$container" >/dev/null 2>&1 || true
  fi
  if [[ -n "$logs_pid" ]] && kill -0 "$logs_pid" 2>/dev/null; then
    wait "$logs_pid" 2>/dev/null || true
  fi
  docker rm "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "mode=$mode"
  echo "kernel_substring=${kernel_substring:-<all-modules>}"
  echo "model=$model"
  echo "revision=$revision"
  echo "image=$image"
  echo "seed=$seed"
  echo "semantic_measurement_layer=true"
  echo "warmup_prompts=$warmup_prompts"
  echo "warmup_output_tokens=$warmup_output_tokens"
  echo "measured_prompts=$measured_prompts"
  echo "measured_output_tokens=$measured_output_tokens"
  echo "max_concurrency=$max_concurrency"
  echo "event_capacity=$event_capacity"
  echo "event_size_bytes=64"
  echo "event_buffer_bytes=$((event_capacity * 64))"
  echo "sample_log2=$sample_log2"
  echo "launch_identity=$launch_identity"
  uname -a
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,power.draw --format=csv,noheader
  docker image inspect "$image" --format 'image_id={{.Id}} image_digests={{json .RepoDigests}}'
  sha256sum "$semantic_lib" "$semantic_core" "$semantic_py" \
            "$sanitizer_build/libgpu_observer_sanitizer.so" \
            "$sanitizer_build/gpu_observer_sanitizer_patches.cubin"
} > "$run_dir/manifest.txt"

docker_args=(
  run -d
  --name "$container"
  --gpus all
  --ipc=host
  --network=host
  --security-opt label=disable
  -e GPU_OBSERVER_SEMANTIC_LIB=/observer/libgpu_observer_vllm_bridge.so
  -e GPU_OBSERVER_SEMANTIC_SHM=/observer-shm/semantic.ring
  -e GPU_OBSERVER_SEMANTIC_CAPACITY=65536
  -e GPU_OBSERVER_MAX_SLICES=1024
  -v /home/harsh4786/.cache/huggingface:/root/.cache/huggingface
  -v "$semantic_lib:/observer/libgpu_observer_vllm_bridge.so:ro"
  -v "$semantic_core:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py:ro"
  -v "$semantic_py:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/gpu_observer_semantic.py:ro"
  -v "$run_dir/semantic-shm:/observer-shm"
  -v "$run_dir:/results"
)

if [[ "$mode" != clean ]]; then
  docker_args+=(
    -e LD_PRELOAD=/observer-sanitizer/libgpu_observer_sanitizer.so
    -e GPU_OBSERVER_SAN_MODE="$mode"
    -e GPU_OBSERVER_SAN_PATCH_FILE=/observer-sanitizer/gpu_observer_sanitizer_patches.cubin
    -e GPU_OBSERVER_SAN_OUTPUT_PREFIX=/results/device-probe
    -e GPU_OBSERVER_SAN_EVENT_CAPACITY="$event_capacity"
    -e GPU_OBSERVER_SAN_SAMPLE_LOG2="$sample_log2"
    -v "$sanitizer_build:/observer-sanitizer:ro"
  )
  if [[ -n "$kernel_substring" ]]; then
    docker_args+=( -e GPU_OBSERVER_SAN_KERNEL_SUBSTRING="$kernel_substring" )
  fi
  if [[ "$launch_identity" == 1 ]]; then
    [[ -n "$kernel_substring" ]] || {
      echo "GO_SAN_LAUNCH_ID=1 requires KERNEL_SUBSTRING" >&2; exit 2;
    }
    docker_args+=( -e GPU_OBSERVER_SAN_LAUNCH_ID=1 )
  fi
fi

server_args=(
  vllm serve "$model"
  --revision "$revision"
  --port "$port"
  --dtype bfloat16
  --max-model-len 4096
  --kv-cache-memory-bytes 8G
  --max-num-seqs 32
  --enforce-eager
  --no-async-scheduling
  --no-enable-prefix-caching
)

printf '%q ' docker "${docker_args[@]}" "$image" "${server_args[@]}" > "$run_dir/server-command.txt"
printf '\n' >> "$run_dir/server-command.txt"
docker "${docker_args[@]}" "$image" "${server_args[@]}" > "$run_dir/container-id.txt"
docker logs -f "$container" > "$run_dir/server.log" 2>&1 &
logs_pid=$!

ready=0
for _ in $(seq 1 900); do
  if curl -fsS -o /dev/null "http://127.0.0.1:$port/health" 2>/dev/null; then
    ready=1
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -Fxq "$container"; then
    echo "server exited before ready" >&2
    tail -200 "$run_dir/server.log" >&2
    exit 1
  fi
  sleep 1
done
[[ "$ready" -eq 1 ]] || { echo "server readiness timeout" >&2; exit 1; }

docker exec "$container" chmod 0666 /observer-shm/semantic.ring
engine_host_pid=$(docker top "$container" -eo pid,comm,args | awk '$2 ~ /^VLLM::EngineCor/ {print $1; exit}')
[[ -n "$engine_host_pid" ]] || { echo "could not resolve EngineCore host PID" >&2; exit 1; }
echo "$engine_host_pid" > "$run_dir/engine-host-pid.txt"
docker top "$container" -eo pid,comm,args > "$run_dir/process-topology.txt"

run_client() {
  local prompts=$1
  local output_tokens=$2
  local filename=$3
  local label=$4
  local log_file=$5
  local command=(
    docker run --rm --network host
    -v /home/harsh4786/.cache/huggingface:/root/.cache/huggingface:ro
    -v "$run_dir:/results"
    --entrypoint vllm
    "$image"
    bench serve
    --backend openai
    --base-url "http://127.0.0.1:$port"
    --endpoint /v1/completions
    --model "$model"
    --dataset-name random
    --random-input-len 128
    --random-output-len "$output_tokens"
    --random-range-ratio 0.0
    --num-prompts "$prompts"
    --max-concurrency "$max_concurrency"
    --request-rate inf
    --ignore-eos
    --temperature 0
    --seed "$seed"
    --metric-percentiles 50,95,99
    --percentile-metrics ttft,tpot,itl,e2el
    --save-result
    --save-detailed
    --result-dir /results
    --result-filename "$filename"
    --label "$label"
    --request-id-prefix "${mode}-${label}-"
  )
  if [[ "$label" == measured ]]; then
    printf '%q ' "${command[@]}" > "$run_dir/benchmark-command.txt"
    printf '\n' >> "$run_dir/benchmark-command.txt"
  fi
  "${command[@]}" > "$log_file" 2>&1
}

# Warm all normal inference paths, then remove warmup records from the semantic ring.
if [[ "$warmup_prompts" -gt 0 ]]; then
  run_client "$warmup_prompts" "$warmup_output_tokens" \
    warmup.json warmup "$run_dir/warmup-client.log"
  "$semantic_capture" "$run_dir/semantic-shm/semantic.ring" \
    "$run_dir/warmup-semantic.bin" 1 > "$run_dir/warmup-semantic.log" 2>&1
else
  echo "warmup skipped" > "$run_dir/warmup-client.log"
fi

if [[ "$mode" != clean && "$launch_identity" == 1 ]]; then
  arm_pid=$(docker exec "$container" bash -lc \
    "ps -eo pid,comm | awk '\$2 ~ /^VLLM::EngineCor/ {print \$1; exit}'")
  [[ -n "$arm_pid" ]] || { echo "could not resolve EngineCore PID for capture arm" >&2; exit 1; }
  docker exec "$container" kill -USR2 "$arm_pid"
  echo "armed_after_warmup=true" >> "$run_dir/manifest.txt"
fi

{
  echo 'realtime_ns,total_jiffies,idle_jiffies,mem_total_kb,mem_available_kb,engine_cpu_pct,engine_rss_kb,engine_threads,gpu_util_pct,gpu_memory_mib,gpu_temp_c,gpu_power_w'
  while docker ps --format '{{.Names}}' | grep -Fxq "$container"; do
    realtime_ns=$(date +%s%N)
    read -r _ cpu_user cpu_nice cpu_system cpu_idle cpu_iowait cpu_irq cpu_softirq cpu_steal _ < /proc/stat
    total_jiffies=$((cpu_user + cpu_nice + cpu_system + cpu_idle + cpu_iowait + cpu_irq + cpu_softirq + cpu_steal))
    idle_jiffies=$((cpu_idle + cpu_iowait))
    read -r mem_total_kb mem_available_kb < <(awk '/^MemTotal:/ {total=$2} /^MemAvailable:/ {available=$2} END {print total, available}' /proc/meminfo)
    read -r engine_cpu_pct engine_rss_kb engine_threads < <(ps -p "$engine_host_pid" -o %cpu=,rss=,nlwp= | awk 'NF == 3 {print $1,$2,$3}')
    gpu_csv=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits | head -1 | tr -d ' ')
    IFS=',' read -r gpu_util_pct gpu_memory_mib gpu_temp_c gpu_power_w <<< "$gpu_csv"
    echo "$realtime_ns,$total_jiffies,$idle_jiffies,$mem_total_kb,$mem_available_kb,${engine_cpu_pct:-NA},${engine_rss_kb:-NA},${engine_threads:-NA},${gpu_util_pct:-NA},${gpu_memory_mib:-NA},${gpu_temp_c:-NA},${gpu_power_w:-NA}"
    sleep 1
  done
} > "$run_dir/system.csv" &
monitor_pid=$!

run_client "$measured_prompts" "$measured_output_tokens" \
  client.json measured "$run_dir/client.log"

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=

"$semantic_capture" "$run_dir/semantic-shm/semantic.ring" \
  "$run_dir/semantic.bin" 1 > "$run_dir/semantic-capture.log" 2>&1
"$semantic_stats" "$run_dir/semantic.bin" > "$run_dir/engine-step-stats.tsv"

if [[ "$mode" != clean ]]; then
  engine_container_pid=$(docker exec "$container" bash -lc \
    "ps -eo pid,comm | awk '\$2 ~ /^VLLM::EngineCor/ {print \$1; exit}'")
  [[ -n "$engine_container_pid" ]] || { echo "could not resolve EngineCore container PID" >&2; exit 1; }
  docker exec "$container" kill -USR2 "$engine_container_pid"
  curl -fsS -o "$run_dir/flush-response.json" \
    -H 'Content-Type: application/json' \
    --data "{\"model\":\"$model\",\"prompt\":\"flush probe snapshot\",\"max_tokens\":1,\"temperature\":0}" \
    "http://127.0.0.1:$port/v1/completions"
  for _ in $(seq 1 100); do
    [[ -s "$run_dir/device-probe.summary.tsv" ]] && break
    sleep 0.1
  done
  [[ -s "$run_dir/device-probe.summary.tsv" ]] || { echo "sanitizer probe did not flush" >&2; exit 1; }
  if [[ "$launch_identity" == 1 ]]; then
    [[ -x "$join_b_cache" ]] || { echo "missing Join B validator: $join_b_cache" >&2; exit 1; }
    "$join_b_cache" \
      "$run_dir/semantic.bin" \
      "$run_dir/device-probe.launches.tsv" \
      "$run_dir/device-probe.events.bin" > "$run_dir/join-b-cache.log"
  fi
fi

jq -r '
  ["metric","value","unit"],
  ["completed",.completed,"requests"],
  ["failed",.failed,"requests"],
  ["request_throughput",.request_throughput,"requests_per_second"],
  ["output_throughput",.output_throughput,"tokens_per_second"],
  ["total_token_throughput",.total_token_throughput,"tokens_per_second"],
  ["ttft_p50",.p50_ttft_ms,"ms"],
  ["ttft_p95",.p95_ttft_ms,"ms"],
  ["ttft_p99",.p99_ttft_ms,"ms"],
  ["itl_p50",.p50_itl_ms,"ms"],
  ["itl_p95",.p95_itl_ms,"ms"],
  ["itl_p99",.p99_itl_ms,"ms"],
  ["e2e_p50",.p50_e2el_ms,"ms"],
  ["e2e_p95",.p95_e2el_ms,"ms"],
  ["e2e_p99",.p99_e2el_ms,"ms"] | @tsv
' "$run_dir/client.json" > "$run_dir/client-metrics.tsv"

awk -F, '
  NR == 1 {next}
  NR == 2 {first_total=$2; first_idle=$3}
  {
    last_total=$2; last_idle=$3; samples++;
    used_kb=$4-$5; used_sum+=used_kb; if (used_kb>used_max) used_max=used_kb;
    if ($6 != "NA") {cpu_sum+=$6; cpu_n++}
    if ($7 != "NA" && $7>rss_max) rss_max=$7;
    if ($9 != "NA") {gpu_sum+=$9; gpu_n++; if ($9>gpu_max) gpu_max=$9}
    if ($11 != "NA" && $11>temp_max) temp_max=$11;
    if ($12 != "NA") {power_sum+=$12; power_n++; if ($12>power_max) power_max=$12}
  }
  END {
    cpu_total_util=(last_total>first_total) ? 100*(1-(last_idle-first_idle)/(last_total-first_total)) : 0;
    print "metric\tvalue\tunit";
    print "samples\t" samples "\tcount";
    print "host_cpu_mean\t" cpu_total_util "\tpercent";
    print "engine_cpu_mean\t" (cpu_n?cpu_sum/cpu_n:0) "\tpercent_of_one_core";
    print "engine_rss_max\t" rss_max "\tKiB";
    print "system_memory_used_mean\t" (samples?used_sum/samples:0) "\tKiB";
    print "system_memory_used_max\t" used_max "\tKiB";
    print "gpu_util_mean\t" (gpu_n?gpu_sum/gpu_n:0) "\tpercent";
    print "gpu_util_max\t" gpu_max "\tpercent";
    print "gpu_temp_max\t" temp_max "\tC";
    print "gpu_power_mean\t" (power_n?power_sum/power_n:0) "\tW";
    print "gpu_power_max\t" power_max "\tW";
  }
' "$run_dir/system.csv" > "$run_dir/system-summary.tsv"

docker stop --time 30 "$container" >/dev/null
wait "$logs_pid" 2>/dev/null || true
logs_pid=
docker rm "$container" >/dev/null

{
  echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader
} > "$run_dir/final-state.txt"

chmod -R a+rX "$run_dir" 2>/dev/null || true
"$root/benchmarks/seal-run.sh" "$run_dir"
trap - EXIT INT TERM

echo "completed rung=$mode artifacts=$run_dir"

