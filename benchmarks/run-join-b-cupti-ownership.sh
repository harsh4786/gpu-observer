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
join_b="$root/target/release/join_b_cupti_cache"
semantic_core="$root/vllm-adapter/overlay/vllm/v1/engine/core.py"
semantic_py="$root/vllm-adapter/gpu_observer_semantic.py"
packing_runner="$root/vllm-adapter/packed-overlay/vllm/v1/worker/gpu_model_runner.py"
cupti_lib="$root/cupti-agent/activity/build-cuda1321/libgpu_observer_cupti_activity.so"
port=8000
expected_nodes=40
output_lengths=(8 12 16 20 24 28 32 36)
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$root/benchmarks/join-b/exp0020-cupti-ownership/$stamp"
container="go-exp0020-$stamp"

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
                "$packing_runner" "$cupti_lib"; do
  [[ -f "$required" ]] || { echo "missing required artifact: $required" >&2; exit 1; }
done

{
  echo "experiment=EXP-0020-cupti-ownership"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "question=Can CUPTI correlation IDs and actual GPU intervals replace ordered-count attribution while preserving authoritative packed-row ownership?"
  echo "model=$model"
  echo "revision=$revision"
  echo "image=$image"
  echo "execution_mode=FULL_AND_PIECEWISE_CUDA_GRAPHS"
  echo "async_scheduling=production_default"
  echo "prefix_caching=false"
  echo "target_kernel=reshape_and_cache_flash_kernel"
  echo "observer_semantic=vLLM_patched_CLOCK_MONOTONIC"
  echo "observer_gpu=CUPTI_CONCURRENT_KERNEL_actual_start_end"
  echo "observer_cuda_api=CUPTI_RUNTIME_activity_correlation_id"
  echo "clock_mapping=two_point_affine_CUPTI_to_CLOCK_MONOTONIC"
  echo "cupti_buffer_count=8"
  echo "cupti_buffer_bytes=1048576"
  echo "cupti_total_buffer_bytes=8388608"
  echo "cupti_runtime_record_capacity=65536"
  echo "device_internal_events=absent"
  echo "device_internal_reason=Compute_Sanitizer_and_direct_CUPTI_are_mutually_exclusive_subscribers"
  echo "ownership_evidence=authoritative_packed_rows_plus_CUPTI_launch_geometry"
  echo "request_count=${#output_lengths[@]}"
  echo "output_lengths=${output_lengths[*]}"
  echo "ignore_eos=true"
  echo "expected_target_kernels_per_step=$expected_nodes"
  echo "semantic_abi_version=2"
  uname -a
  free -b
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,power.draw --format=csv,noheader
  docker image inspect "$image" --format 'image_id={{.Id}} image_digests={{json .RepoDigests}}'
  sha256sum "$payloads" "$semantic_lib" "$semantic_core" "$semantic_py" \
            "$packing_runner" "$cupti_lib" "$join_b"
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
  -e LD_PRELOAD=/observer-cupti/libgpu_observer_cupti_activity.so \
  -e GPU_OBSERVER_CUPTI_DEFER=1 \
  -e GPU_OBSERVER_CUPTI_OUTPUT_PREFIX=/results/cupti-%p \
  -e GPU_OBSERVER_CUPTI_KERNEL_SUBSTRING=reshape_and_cache_flash_kernel \
  -v /home/harsh4786/.cache/huggingface:/root/.cache/huggingface \
  -v "$semantic_lib:/observer/libgpu_observer_vllm_bridge.so:ro" \
  -v "$semantic_core:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py:ro" \
  -v "$semantic_py:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/gpu_observer_semantic.py:ro" \
  -v "$packing_runner:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py:ro" \
  -v "$cupti_lib:/observer-cupti/libgpu_observer_cupti_activity.so:ro" \
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
[[ "$engine_pid" =~ ^[0-9]+$ ]] || { echo "could not resolve EngineCore PID" >&2; exit 1; }
echo "$engine_pid" > "$run_dir/engine-container-pid.txt"
docker top "$container" -eo pid,comm,args > "$run_dir/process-topology.txt"
rg -F "Asynchronous scheduling is enabled" "$run_dir/server.log" > "$run_dir/async-scheduling-evidence.txt" || true
rg -F "FULL_AND_PIECEWISE" "$run_dir/server.log" > "$run_dir/cuda-graph-evidence.txt" || true
docker exec -e LD_PRELOAD= "$container" \
  test -p "/tmp/gpu-observer-cupti-$engine_pid.fifo"
printf 'engine_pid=%s control=/tmp/gpu-observer-cupti-%s.fifo\n' \
  "$engine_pid" "$engine_pid" > "$run_dir/probe-init-evidence.txt"

head -n 1 "$payloads" | jq \
  '.stream=false | .max_completion_tokens=2 | .ignore_eos=true | .temperature=0' \
  > "$run_dir/warmup-request.json"
curl -fsS --max-time 300 \
  -H 'Content-Type: application/json' \
  -H 'X-Request-Id: exp0020-warmup' \
  --data-binary @"$run_dir/warmup-request.json" \
  "http://127.0.0.1:$port/v1/chat/completions" \
  -o "$run_dir/warmup-response.json"
"$semantic_capture" "$run_dir/semantic-shm/semantic.ring" \
  "$run_dir/warmup-semantic.bin" 1 > "$run_dir/warmup-semantic.log" 2>&1

docker exec -e LD_PRELOAD= "$container" sh -c \
  "printf S > /tmp/gpu-observer-cupti-$engine_pid.fifo"
cupti_started=0
for _ in $(seq 1 200); do
  if rg -Fq "gpu-observer-cupti: control command=S status=0" "$run_dir/server.log"; then
    cupti_started=1
    break
  fi
  sleep 0.1
done
[[ "$cupti_started" == 1 ]] || { echo "CUPTI deferred start failed" >&2; exit 1; }

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
  run_request "$request_number" "exp0020-request-$request_number" \
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

docker exec -e LD_PRELOAD= "$container" sh -c \
  "printf F > /tmp/gpu-observer-cupti-$engine_pid.fifo"
cupti_activity="$run_dir/cupti-$engine_pid.activities.tsv"
cupti_runtime="$run_dir/cupti-$engine_pid.runtime.tsv"
cupti_summary="$run_dir/cupti-$engine_pid.summary.tsv"
cupti_flushed=0
for _ in $(seq 1 300); do
  if [[ -s "$cupti_summary" && -s "$cupti_activity" && -s "$cupti_runtime" ]] \
     && rg -Fq "gpu-observer-cupti: control command=F status=0" "$run_dir/server.log"; then
    cupti_flushed=1
    break
  fi
  sleep 0.1
done
[[ "$cupti_flushed" == 1 ]] || { echo "CUPTI flush failed" >&2; exit 1; }
for required in "$cupti_summary" "$cupti_activity" "$cupti_runtime"; do
  [[ -s "$required" ]] || { echo "missing CUPTI output: $required" >&2; exit 1; }
done

set +e
"$join_b" \
  "$run_dir/semantic.bin" \
  "$cupti_activity" \
  "$cupti_runtime" \
  "$cupti_summary" \
  --expected-kernels "$expected_nodes" \
  --decode-suffix \
  --require-reordering \
  > "$run_dir/join-b-cupti.log" 2> "$run_dir/join-b-cupti.stderr"
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
