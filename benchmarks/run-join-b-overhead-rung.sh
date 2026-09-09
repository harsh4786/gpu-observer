#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 EXPERIMENT_ROOT {clean|scheduler|packed|sass|cupti} REPEAT" >&2
  exit 2
}

[[ $# -eq 3 ]] || usage
experiment_root=$(realpath -m "$1")
arm=$2
repeat=$3
case "$arm" in
  clean|scheduler|packed|sass|cupti) ;;
  *) usage ;;
esac
[[ "$repeat" =~ ^[1-9][0-9]*$ ]] || usage

root=/home/harsh4786/gpu-observer
image=gpu-observer/vllm-sanitizer:cuda13.2
model=Qwen/Qwen3-14B
revision=40c069824f4251a91eefaf281ebe4c544efd3e18
port=8000
seed=20260812
warmup_prompts=8
warmup_output_tokens=16
measured_prompts=64
measured_input_tokens=128
measured_output_tokens=64
max_concurrency=8
semantic_capacity=65536
max_slices=1024
event_capacity=1048576
target_kernel=reshape_and_cache_flash_kernel
run_dir="$experiment_root/repeat-$repeat/$arm"
container="go-joinb-oh-${arm}-r${repeat}-$(date -u +%H%M%S)"

if [[ -e "$run_dir" ]]; then
  echo "refusing to overwrite existing rung: $run_dir" >&2
  exit 2
fi
mkdir -p "$run_dir"
chmod 0777 "$run_dir"

semantic_lib="$root/target/release/libgpu_observer_vllm_bridge.so"
semantic_capture="$root/target/release/semantic_capture"
semantic_stats="$root/target/release/semantic_step_stats"
join_b="$root/target/release/join_b_cache"
semantic_core="$root/vllm-adapter/overlay/vllm/v1/engine/core.py"
semantic_py="$root/vllm-adapter/gpu_observer_semantic.py"
packed_runner="$root/vllm-adapter/packed-overlay/vllm/v1/worker/gpu_model_runner.py"
attention_observer="$root/vllm-adapter/gpu_observer_attention.py"
san_build="$root/device-probes/compute-sanitizer/build"
cupti_lib="$root/cupti-agent/activity/build-cuda1321/libgpu_observer_cupti_activity.so"

semantic_enabled=0
packed_enabled=0
sass_enabled=0
cupti_enabled=0
case "$arm" in
  scheduler) semantic_enabled=1 ;;
  packed) semantic_enabled=1; packed_enabled=1 ;;
  sass) semantic_enabled=1; packed_enabled=1; sass_enabled=1 ;;
  cupti) semantic_enabled=1; packed_enabled=1; cupti_enabled=1 ;;
esac

# CUPTI and Compute Sanitizer cannot share a process on this stack (one CUPTI
# subscriber slot), so the sass and cupti arms are alternatives, never combined.
if [[ "$sass_enabled" == 1 && "$cupti_enabled" == 1 ]]; then
  echo "sass and cupti arms are mutually exclusive" >&2
  exit 2
fi

if [[ "$semantic_enabled" == 1 ]]; then
  mkdir -p "$run_dir/semantic-shm"
  chmod 0777 "$run_dir/semantic-shm"
  for required in "$semantic_lib" "$semantic_capture" "$semantic_stats" \
                  "$semantic_core" "$semantic_py"; do
    [[ -f "$required" ]] || { echo "missing required artifact: $required" >&2; exit 1; }
  done
fi
if [[ "$packed_enabled" == 1 ]]; then
  [[ -f "$packed_runner" ]] || { echo "missing packed runner: $packed_runner" >&2; exit 1; }
  # The packed overlay imports vllm.gpu_observer_attention; the hook is inert
  # unless GPU_OBSERVER_ATTENTION_TENSOR_RANGES is set, but the module must be
  # importable or EngineCore dies at gpu_model_runner import time.
  [[ -f "$attention_observer" ]] || { echo "missing attention observer: $attention_observer" >&2; exit 1; }
fi
if [[ "$sass_enabled" == 1 ]]; then
  for required in "$join_b" "$san_build/libgpu_observer_sanitizer.so" \
                  "$san_build/gpu_observer_sanitizer_patches.cubin"; do
    [[ -f "$required" ]] || { echo "missing required artifact: $required" >&2; exit 1; }
  done
fi
if [[ "$cupti_enabled" == 1 ]]; then
  [[ -f "$cupti_lib" ]] || { echo "missing required artifact: $cupti_lib" >&2; exit 1; }
fi

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
  echo "experiment=EXP-0015"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "arm=$arm"
  echo "repeat=$repeat"
  echo "model=$model"
  echo "revision=$revision"
  echo "image=$image"
  echo "execution_mode=eager"
  echo "async_scheduling=false"
  echo "prefix_caching=false"
  echo "seed=$seed"
  echo "warmup_prompts=$warmup_prompts"
  echo "measured_prompts=$measured_prompts"
  echo "measured_input_tokens=$measured_input_tokens"
  echo "measured_output_tokens=$measured_output_tokens"
  echo "max_concurrency=$max_concurrency"
  echo "semantic_enabled=$semantic_enabled"
  echo "packed_enabled=$packed_enabled"
  echo "sass_enabled=$sass_enabled"
  echo "cupti_enabled=$cupti_enabled"
  echo "semantic_capacity=$semantic_capacity"
  echo "semantic_record_bytes=96"
  echo "semantic_buffer_bytes=$((semantic_capacity * 96 + 256))"
  echo "max_packed_slices=$max_slices"
  echo "packed_ctypes_bytes=$((max_slices * 40))"
  echo "sass_target_kernel=$target_kernel"
  echo "sass_event_capacity=$event_capacity"
  echo "sass_event_bytes=64"
  echo "sass_event_buffer_bytes=$((event_capacity * 64))"
  echo "cupti_kernel_filter=none_unfiltered"
  echo "cupti_defer=1"
  uname -a
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,power.draw --format=csv,noheader
  docker image inspect "$image" --format 'image_id={{.Id}} image_digests={{json .RepoDigests}}'
  if [[ "$semantic_enabled" == 1 ]]; then
    sha256sum "$semantic_lib" "$semantic_core" "$semantic_py"
  fi
  if [[ "$packed_enabled" == 1 ]]; then
    sha256sum "$packed_runner" "$attention_observer"
  fi
  if [[ "$sass_enabled" == 1 ]]; then
    sha256sum "$san_build/libgpu_observer_sanitizer.so" \
              "$san_build/gpu_observer_sanitizer_patches.cubin" "$join_b"
  fi
  if [[ "$cupti_enabled" == 1 ]]; then
    sha256sum "$cupti_lib"
  fi
} > "$run_dir/manifest.txt"

docker_args=(
  run -d
  --name "$container"
  --gpus all
  --ipc=host
  --network=host
  --security-opt label=disable
  -v /home/harsh4786/.cache/huggingface:/root/.cache/huggingface
  -v "$run_dir:/results"
)

if [[ "$semantic_enabled" == 1 ]]; then
  docker_args+=(
    -e GPU_OBSERVER_SEMANTIC_LIB=/observer/libgpu_observer_vllm_bridge.so
    -e GPU_OBSERVER_SEMANTIC_SHM=/observer-shm/semantic.ring
    -e GPU_OBSERVER_SEMANTIC_CAPACITY="$semantic_capacity"
    -e GPU_OBSERVER_MAX_SLICES="$max_slices"
    -v "$semantic_lib:/observer/libgpu_observer_vllm_bridge.so:ro"
    -v "$semantic_core:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py:ro"
    -v "$semantic_py:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/gpu_observer_semantic.py:ro"
    -v "$run_dir/semantic-shm:/observer-shm"
  )
fi

if [[ "$packed_enabled" == 1 ]]; then
  docker_args+=(
    -v "$packed_runner:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py:ro"
    -v "$attention_observer:/usr/local/lib/python3.12/dist-packages/vllm/gpu_observer_attention.py:ro"
  )
fi

if [[ "$sass_enabled" == 1 ]]; then
  docker_args+=(
    -e LD_PRELOAD=/observer-sanitizer/libgpu_observer_sanitizer.so
    -e GPU_OBSERVER_SAN_MODE=block_event
    -e GPU_OBSERVER_SAN_PATCH_FILE=/observer-sanitizer/gpu_observer_sanitizer_patches.cubin
    -e GPU_OBSERVER_SAN_OUTPUT_PREFIX=/results/device-probe
    -e GPU_OBSERVER_SAN_EVENT_CAPACITY="$event_capacity"
    -e GPU_OBSERVER_SAN_KERNEL_SUBSTRING="$target_kernel"
    -e GPU_OBSERVER_SAN_LAUNCH_ID=1
    -v "$san_build:/observer-sanitizer:ro"
  )
fi

# Deliberately UNFILTERED (no GPU_OBSERVER_CUPTI_KERNEL_SUBSTRING), matching
# run-live-demo-cupti.sh -- the point of this arm is to price the configuration
# the live demo actually runs, which captures every kernel launch.
if [[ "$cupti_enabled" == 1 ]]; then
  docker_args+=(
    -e LD_PRELOAD=/observer-cupti/libgpu_observer_cupti_activity.so
    -e GPU_OBSERVER_CUPTI_DEFER=1
    -e GPU_OBSERVER_CUPTI_OUTPUT_PREFIX=/results/cupti-%p
    -v "$cupti_lib:/observer-cupti/libgpu_observer_cupti_activity.so:ro"
  )
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
[[ "$ready" == 1 ]] || { echo "server readiness timeout" >&2; exit 1; }

if [[ "$sass_enabled" == 1 ]]; then
  expected_init="initialized mode=block_event capacity=$event_capacity"
  if ! grep -Fq "$expected_init" "$run_dir/server.log"; then
    echo "SASS subscriber did not accept requested event capacity: $event_capacity" >&2
    grep -F 'gpu-observer-sanitizer: initialized' "$run_dir/server.log" >&2 || true
    exit 1
  fi
fi

if [[ "$semantic_enabled" == 1 ]]; then
  docker exec "$container" chmod 0666 /observer-shm/semantic.ring
fi
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
    --random-input-len "$measured_input_tokens"
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
    --label "${arm}-r${repeat}-${label}"
    --request-id-prefix "exp0015-${arm}-r${repeat}-${label}-"
  )
  if [[ "$label" == measured ]]; then
    printf '%q ' "${command[@]}" > "$run_dir/benchmark-command.txt"
    printf '\n' >> "$run_dir/benchmark-command.txt"
  fi
  "${command[@]}" > "$log_file" 2>&1
}

run_client "$warmup_prompts" "$warmup_output_tokens" \
  warmup.json warmup "$run_dir/warmup-client.log"
if [[ "$semantic_enabled" == 1 ]]; then
  "$semantic_capture" "$run_dir/semantic-shm/semantic.ring" \
    "$run_dir/warmup-semantic.bin" 1 > "$run_dir/warmup-semantic.log" 2>&1
fi

if [[ "$sass_enabled" == 1 ]]; then
  engine_container_pid=$(docker exec "$container" bash -lc \
    "ps -eo pid,comm | awk '\$2 ~ /^VLLM::EngineCor/ {print \$1; exit}'")
  [[ -n "$engine_container_pid" ]] || { echo "could not resolve EngineCore PID" >&2; exit 1; }
  docker exec "$container" kill -USR2 "$engine_container_pid"
  echo "sass_capture_armed_after_warmup=true" >> "$run_dir/manifest.txt"
fi

if [[ "$cupti_enabled" == 1 ]]; then
  engine_container_pid=$(docker exec "$container" bash -lc \
    "ps -eo pid,comm | awk '\$2 ~ /^VLLM::EngineCor/ {print \$1; exit}'")
  [[ "$engine_container_pid" =~ ^[0-9]+$ ]] || { echo "could not resolve EngineCore PID" >&2; exit 1; }
  echo "$engine_container_pid" > "$run_dir/engine-container-pid.txt"
  cupti_fifo="/tmp/gpu-observer-cupti-$engine_container_pid.fifo"
  docker exec -e LD_PRELOAD= "$container" test -p "$cupti_fifo"
  docker exec -e LD_PRELOAD= "$container" sh -c "printf S > $cupti_fifo"
  cupti_started=0
  for _ in $(seq 1 100); do
    if grep -Fq "gpu-observer-cupti: control command=S status=0" "$run_dir/server.log"; then
      cupti_started=1
      break
    fi
    sleep 0.1
  done
  [[ "$cupti_started" == 1 ]] || { echo "CUPTI deferred start failed" >&2; tail -50 "$run_dir/server.log" >&2; exit 1; }
  echo "cupti_capture_armed_after_warmup=true" >> "$run_dir/manifest.txt"
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
    gpu_csv=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)
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

if [[ "$semantic_enabled" == 1 ]]; then
  "$semantic_capture" "$run_dir/semantic-shm/semantic.ring" \
    "$run_dir/semantic.bin" 1 > "$run_dir/semantic-capture.log" 2>&1
  "$semantic_stats" "$run_dir/semantic.bin" > "$run_dir/engine-step-stats.tsv"
fi

if [[ "$sass_enabled" == 1 ]]; then
  docker exec "$container" kill -USR2 "$engine_container_pid"
  curl -fsS -o "$run_dir/flush-response.json" \
    -H 'Content-Type: application/json' \
    --data "{\"model\":\"$model\",\"prompt\":\"flush probe snapshot\",\"max_tokens\":1,\"temperature\":0}" \
    "http://127.0.0.1:$port/v1/completions"
  for _ in $(seq 1 100); do
    [[ -s "$run_dir/device-probe.summary.tsv" ]] && break
    sleep 0.1
  done
  [[ -s "$run_dir/device-probe.summary.tsv" ]] || { echo "device snapshot missing" >&2; exit 1; }
  "$join_b" "$run_dir/semantic.bin" "$run_dir/device-probe.launches.tsv" \
    "$run_dir/device-probe.events.bin" --require-packed > "$run_dir/join-b-cache.log"
fi

if [[ "$cupti_enabled" == 1 ]]; then
  docker exec -e LD_PRELOAD= "$container" sh -c "printf F > $cupti_fifo"
  cupti_activity="$run_dir/cupti-$engine_container_pid.activities.tsv"
  cupti_runtime="$run_dir/cupti-$engine_container_pid.runtime.tsv"
  cupti_summary="$run_dir/cupti-$engine_container_pid.summary.tsv"
  for _ in $(seq 1 300); do
    if [[ -s "$cupti_summary" && -s "$cupti_activity" && -s "$cupti_runtime" ]] \
      && grep -Fq "gpu-observer-cupti: control command=F status=0" "$run_dir/server.log"; then
      break
    fi
    sleep 0.1
  done
  for required in "$cupti_summary" "$cupti_activity" "$cupti_runtime"; do
    [[ -s "$required" ]] || { echo "missing CUPTI output: $required" >&2; exit 1; }
  done
  echo "cupti_kernel_rows=$(($(wc -l < "$cupti_activity") - 1))" >> "$run_dir/manifest.txt"
fi

jq -r '
  ["metric","value","unit"],
  ["completed",.completed,"requests"],
  ["failed",.failed,"requests"],
  ["duration",.duration,"seconds"],
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
    used=$4-$5; used_sum+=used; if (used>used_max) used_max=used;
    if ($6 != "NA") {engine_cpu_sum+=$6; engine_cpu_n++}
    if ($7 != "NA" && $7>rss_max) rss_max=$7;
    if ($9 != "NA" && $9 != "[N/A]") {gpu_sum+=$9; gpu_n++; if ($9>gpu_max) gpu_max=$9}
    if ($11 != "NA" && $11>temp_max) temp_max=$11;
    if ($12 != "NA") {power_sum+=$12; power_n++; if ($12>power_max) power_max=$12}
  }
  END {
    host_cpu=(last_total>first_total) ? 100*(1-(last_idle-first_idle)/(last_total-first_total)) : 0;
    print "metric\tvalue\tunit";
    print "samples\t" samples "\tcount";
    print "host_cpu_mean\t" host_cpu "\tpercent";
    print "engine_cpu_mean\t" (engine_cpu_n?engine_cpu_sum/engine_cpu_n:0) "\tpercent_of_one_core";
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

docker logs "$container" > "$run_dir/server-final.log" 2>&1 || true
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
echo "completed arm=$arm repeat=$repeat artifacts=$run_dir"
