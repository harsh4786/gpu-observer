#!/usr/bin/env bash
set -euo pipefail

root=/home/harsh4786/gpu-observer
image=gpu-observer/vllm-sanitizer:cuda13.2
model=Qwen/Qwen3-14B
revision=40c069824f4251a91eefaf281ebe4c544efd3e18
payloads="$root/benchmarks/mixed-prefill/exp0012-20260811T143000Z/controlled-workload/interactive-payloads-corrected.jsonl"
semantic_lib="$root/target/release/libgpu_observer_vllm_bridge.so"
semantic_capture="$root/target/release/semantic_capture"
semantic_dump="$root/target/release/semantic_dump"
semantic_stats="$root/target/release/semantic_step_stats"
join_b="$root/target/release/join_b_graph_cache"
semantic_core="$root/vllm-adapter/overlay/vllm/v1/engine/core.py"
semantic_py="$root/vllm-adapter/gpu_observer_semantic.py"
packing_runner="$root/vllm-adapter/packed-overlay/vllm/v1/worker/gpu_model_runner.py"
probe_build="$root/device-probes/compute-sanitizer/build-graph-identity"
probe_lib="$probe_build/libgpu_observer_sanitizer.so"
probe_patch="$probe_build/gpu_observer_sanitizer_patches.cubin"
port=8000
event_capacity=1048576
expected_nodes=40
output_lengths=(8 12 16 20 24 28 32 36)
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$root/benchmarks/join-b/exp0019-graph-ownership/$stamp"
container="go-exp0019-$stamp"

mkdir -p "$run_dir/semantic-shm" "$run_dir/requests" "$run_dir/responses"
chmod 0777 "$run_dir" "$run_dir/semantic-shm" "$run_dir/requests" "$run_dir/responses"

request_jobs=()
logs_pid=
gpu_sampler_pid=
cleanup() {
  for job in "${request_jobs[@]}"; do
    kill "$job" 2>/dev/null || true
  done
  if docker ps --format '{{.Names}}' | grep -Fxq "$container"; then
    docker stop --time 30 "$container" >/dev/null 2>&1 || true
  fi
  if [[ -n "$logs_pid" ]] && kill -0 "$logs_pid" 2>/dev/null; then
    wait "$logs_pid" 2>/dev/null || true
  fi
  if [[ -n "$gpu_sampler_pid" ]] && kill -0 "$gpu_sampler_pid" 2>/dev/null; then
    kill "$gpu_sampler_pid" 2>/dev/null || true
    wait "$gpu_sampler_pid" 2>/dev/null || true
  fi
  docker rm "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

for required in "$payloads" "$semantic_lib" "$semantic_capture" "$semantic_dump" \
                "$semantic_stats" "$join_b" "$semantic_core" "$semantic_py" \
                "$packing_runner" "$probe_lib" "$probe_patch"; do
  [[ -f "$required" ]] || { echo "missing required artifact: $required" >&2; exit 1; }
done

{
  echo "experiment=EXP-0019-graph-ownership"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "question=Can authoritative packed rows plus graph replay/node identity recover per-request block ownership for Qwen3-14B decode after compaction?"
  echo "model=$model"
  echo "revision=$revision"
  echo "image=$image"
  echo "execution_mode=FULL_AND_PIECEWISE_CUDA_GRAPHS"
  echo "async_scheduling=production_default"
  echo "prefix_caching=false"
  echo "target_kernel=reshape_and_cache_flash_kernel"
  echo "observer_semantic=vLLM_patched_CLOCK_MONOTONIC"
  echo "observer_graph=Compute_Sanitizer_graph_node_begin_CLOCK_MONOTONIC"
  echo "observer_device=Compute_Sanitizer_SASS_block_entry_raw_globaltimer"
  echo "observer_ordinary_launch=Compute_Sanitizer_launch_begin_CLOCK_MONOTONIC"
  echo "event_partition_assumption=single_context_stream_order_and_complete_geometry"
  echo "device_window_assumption=explicit_tail_after_final_prefill"
  echo "event_capacity=$event_capacity"
  echo "event_size_bytes=64"
  echo "event_buffer_bytes=$((event_capacity * 64))"
  echo "graph_node_capacity=65536"
  echo "request_count=${#output_lengths[@]}"
  echo "output_lengths=${output_lengths[*]}"
  echo "ignore_eos=true"
  echo "expected_target_nodes_per_replay=$expected_nodes"
  echo "semantic_abi_version=2"
  uname -a
  free -b
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,power.draw --format=csv,noheader
  docker image inspect "$image" --format 'image_id={{.Id}} image_digests={{json .RepoDigests}}'
  sha256sum "$payloads" "$semantic_lib" "$semantic_core" "$semantic_py" \
            "$packing_runner" "$probe_lib" "$probe_patch" "$join_b"
} > "$run_dir/manifest.txt"

docker run -d \
  --name "$container" \
  --gpus all \
  --ipc=host \
  --network=host \
  --security-opt label=disable \
  -e GPU_OBSERVER_SEMANTIC_LIB=/observer/libgpu_observer_vllm_bridge.so \
  -e GPU_OBSERVER_SEMANTIC_SHM=/observer-shm/semantic.ring \
  -e GPU_OBSERVER_SEMANTIC_CAPACITY=65536 \
  -e GPU_OBSERVER_MAX_SLICES=1024 \
  -e LD_PRELOAD=/observer-sanitizer/libgpu_observer_sanitizer.so \
  -e GPU_OBSERVER_SAN_MODE=block_event \
  -e GPU_OBSERVER_SAN_PATCH_FILE=/observer-sanitizer/gpu_observer_sanitizer_patches.cubin \
  -e GPU_OBSERVER_SAN_OUTPUT_PREFIX=/results/device-probe \
  -e GPU_OBSERVER_SAN_EVENT_CAPACITY="$event_capacity" \
  -e GPU_OBSERVER_SAN_KERNEL_SUBSTRING=reshape_and_cache_flash_kernel \
  -e GPU_OBSERVER_SAN_CALLBACK_DATA_SCOPE=function \
  -e GPU_OBSERVER_SAN_GRAPH_NODES=1 \
  -e GPU_OBSERVER_SAN_PASSIVE_LAUNCHES=1 \
  -v /home/harsh4786/.cache/huggingface:/root/.cache/huggingface \
  -v "$semantic_lib:/observer/libgpu_observer_vllm_bridge.so:ro" \
  -v "$semantic_core:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py:ro" \
  -v "$semantic_py:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/gpu_observer_semantic.py:ro" \
  -v "$packing_runner:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py:ro" \
  -v "$probe_build:/observer-sanitizer:ro" \
  -v "$run_dir/semantic-shm:/observer-shm" \
  -v "$run_dir:/results" \
  "$image" \
  vllm serve "$model" \
  --revision "$revision" \
  --port "$port" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --kv-cache-memory-bytes 8G \
  --max-num-seqs 32 \
  --no-enable-prefix-caching > "$run_dir/container-id.txt"

docker logs -f "$container" > "$run_dir/server.log" 2>&1 &
logs_pid=$!
(
  while true; do
    date -u +%Y-%m-%dT%H:%M:%SZ
    nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw --format=csv,noheader,nounits
    sleep 5
  done
) > "$run_dir/gpu-samples.log" 2>&1 &
gpu_sampler_pid=$!

ready=0
for _ in $(seq 1 900); do
  if curl -fsS -o /dev/null "http://127.0.0.1:$port/health" 2>/dev/null; then
    ready=1
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -Fxq "$container"; then
    break
  fi
  sleep 1
done
if [[ "$ready" != 1 ]]; then
  echo "server_ready=false" > "$run_dir/outcome.env"
  docker logs "$container" > "$run_dir/server-final.log" 2>&1 || true
  exit 1
fi

docker exec "$container" chmod 0666 /observer-shm/semantic.ring
engine_pid=$(docker exec "$container" bash -lc \
  "ps -eo pid,comm | awk '\$2 ~ /^VLLM::EngineCor/ {print \$1; exit}'")
[[ -n "$engine_pid" ]] || { echo "could not resolve EngineCore PID" >&2; exit 1; }
echo "$engine_pid" > "$run_dir/engine-container-pid.txt"
docker top "$container" -eo pid,comm,args > "$run_dir/process-topology.txt"
rg -F "Asynchronous scheduling is enabled" "$run_dir/server.log" > "$run_dir/async-scheduling-evidence.txt" || true
rg -F "FULL_AND_PIECEWISE" "$run_dir/server.log" > "$run_dir/cuda-graph-evidence.txt" || true
rg -F "initialized mode=block_event" "$run_dir/server.log" > "$run_dir/probe-init-evidence.txt" || true

head -n 1 "$payloads" | jq \
  '.stream=false | .max_completion_tokens=2 | .ignore_eos=true | .temperature=0' \
  > "$run_dir/warmup-request.json"
curl -fsS --max-time 300 \
  -H 'Content-Type: application/json' \
  -H 'X-Request-Id: exp0019-warmup' \
  --data-binary @"$run_dir/warmup-request.json" \
  "http://127.0.0.1:$port/v1/chat/completions" \
  -o "$run_dir/warmup-response.json"
"$semantic_capture" "$run_dir/semantic-shm/semantic.ring" \
  "$run_dir/warmup-semantic.bin" 1 > "$run_dir/warmup-semantic.log" 2>&1

run_request() {
  local index=$1
  local request_id=$2
  local request_path=$3
  local response_path=$4
  local metrics_path=$5
  set +e
  curl -sS --max-time 900 \
    -H 'Content-Type: application/json' \
    -H "X-Request-Id: $request_id" \
    --data-binary @"$request_path" \
    -o "$response_path" \
    -w 'http_code=%{http_code}\ntime_starttransfer_s=%{time_starttransfer}\ntime_total_s=%{time_total}\n' \
    "http://127.0.0.1:$port/v1/chat/completions" \
    > "$metrics_path" 2> "$run_dir/responses/request-$index.stderr"
  local status=$?
  echo "curl_exit=$status" >> "$metrics_path"
  return "$status"
}

for index in "${!output_lengths[@]}"; do
  request_number=$((index + 1))
  tokens=${output_lengths[$index]}
  sed -n "${request_number}p" "$payloads" | jq --argjson output_tokens "$tokens" \
    '.stream=false | .max_completion_tokens=$output_tokens | .ignore_eos=true | .temperature=0' \
    > "$run_dir/requests/request-$request_number.json"
  run_request "$request_number" "exp0019-request-$request_number" \
    "$run_dir/requests/request-$request_number.json" \
    "$run_dir/responses/request-$request_number.json" \
    "$run_dir/responses/request-$request_number.metrics" &
  request_jobs+=("$!")
done

request_status=0
for job in "${request_jobs[@]}"; do
  wait "$job" || request_status=1
done
request_jobs=()
echo "$request_status" > "$run_dir/request-exit-status.txt"

response_gate=0
for index in "${!output_lengths[@]}"; do
  request_number=$((index + 1))
  expected=${output_lengths[$index]}
  observed=$(jq -r '.usage.completion_tokens // -1' \
    "$run_dir/responses/request-$request_number.json" 2>/dev/null || echo -1)
  http_code=$(awk -F= '$1 == "http_code" {print $2}' \
    "$run_dir/responses/request-$request_number.metrics")
  printf '%s\t%s\t%s\t%s\n' "$request_number" "$expected" "$observed" "$http_code"
  if [[ "$observed" != "$expected" || "$http_code" != 200 ]]; then
    response_gate=1
  fi
done > "$run_dir/response-quality.tsv"

"$semantic_capture" "$run_dir/semantic-shm/semantic.ring" \
  "$run_dir/semantic.bin" 1 > "$run_dir/semantic-capture.log" 2>&1
"$semantic_dump" "$run_dir/semantic.bin" > "$run_dir/semantic-dump.log"
"$semantic_stats" "$run_dir/semantic.bin" > "$run_dir/semantic-step-stats.tsv"

# Function-scope mode treats the first SIGUSR2 as a snapshot request.
docker exec "$container" kill -USR2 "$engine_pid"
curl -fsS --max-time 300 \
  -H 'Content-Type: application/json' \
  --data-binary @"$run_dir/warmup-request.json" \
  "http://127.0.0.1:$port/v1/chat/completions" \
  -o "$run_dir/flush-response.json"
for _ in $(seq 1 200); do
  if [[ -s "$run_dir/device-probe.summary.tsv" \
        && -s "$run_dir/device-probe.events.bin" \
        && -s "$run_dir/device-probe.graph-nodes.tsv" \
        && -s "$run_dir/device-probe.launches.tsv" ]]; then
    break
  fi
  sleep 0.1
done
for required in "$run_dir/device-probe.summary.tsv" "$run_dir/device-probe.events.bin" \
                "$run_dir/device-probe.graph-nodes.tsv" "$run_dir/device-probe.launches.tsv"; do
  [[ -s "$required" ]] || { echo "missing probe output: $required" >&2; exit 1; }
done

set +e
"$join_b" \
  "$run_dir/semantic.bin" \
  "$run_dir/device-probe.graph-nodes.tsv" \
  "$run_dir/device-probe.events.bin" \
  --ordinary-launches "$run_dir/device-probe.launches.tsv" \
  --probe-summary "$run_dir/device-probe.summary.tsv" \
  --expected-nodes "$expected_nodes" \
  --device-tail \
  --decode-suffix \
  --require-reordering \
  > "$run_dir/join-b-graph.log" 2> "$run_dir/join-b-graph.stderr"
join_status=$?
set -e
echo "$join_status" > "$run_dir/join-b-exit-status.txt"

fatal_count=$(rg -i -c 'misaligned address|CUDA error|EngineCore encountered a fatal' \
  "$run_dir/server.log" || true)
fatal_count=${fatal_count:-0}

{
  echo "server_ready=true"
  echo "request_exit_status=$request_status"
  echo "response_gate=$response_gate"
  echo "join_exit_status=$join_status"
  echo "fatal_count=$fatal_count"
  if [[ "$request_status" == 0 && "$response_gate" == 0 \
        && "$join_status" == 0 && "$fatal_count" == 0 ]]; then
    echo "result=pass"
  else
    echo "result=reject"
  fi
  echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$run_dir/outcome.env"

docker logs "$container" > "$run_dir/server-final.log" 2>&1 || true
docker stop --time 30 "$container" >/dev/null
wait "$logs_pid" 2>/dev/null || true
logs_pid=
if [[ -n "$gpu_sampler_pid" ]] && kill -0 "$gpu_sampler_pid" 2>/dev/null; then
  kill "$gpu_sampler_pid" 2>/dev/null || true
  wait "$gpu_sampler_pid" 2>/dev/null || true
fi
gpu_sampler_pid=
docker rm "$container" >/dev/null
{
  nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw --format=csv,noheader
  free -b
} > "$run_dir/final-state.txt"
chmod -R a+rX "$run_dir" 2>/dev/null || true
"$root/benchmarks/seal-run.sh" "$run_dir"
trap - EXIT INT TERM
printf '%s\n' "$run_dir"
