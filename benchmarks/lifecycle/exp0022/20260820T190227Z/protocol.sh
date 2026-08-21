#!/usr/bin/env bash
set -euo pipefail

# Diagnostic causal-timing experiment. This is not an overhead benchmark.
# Four observers use CLOCK_MONOTONIC directly or a measured affine mapping:
# Rust client, vLLM frontend, EngineCore, and CUPTI actual kernel activity.

root=/home/harsh4786/gpu-observer
image=gpu-observer/vllm-sanitizer:cuda13.2
model=Qwen/Qwen3-14B
revision=40c069824f4251a91eefaf281ebe4c544efd3e18
port=8000
external_id=exp0022-request-1
semantic_lib="$root/target/release/libgpu_observer_vllm_bridge.so"
semantic_capture="$root/target/release/semantic_capture"
semantic_dump="$root/target/release/semantic_dump"
semantic_stats="$root/target/release/semantic_step_stats"
full_step="$root/target/release/full_step_cupti"
lifecycle_client="$root/target/release/lifecycle_client"
lifecycle_join="$root/target/release/request_lifecycle"
semantic_core="$root/vllm-adapter/overlay/vllm/v1/engine/core.py"
semantic_py="$root/vllm-adapter/gpu_observer_semantic.py"
packing_runner="$root/vllm-adapter/packed-overlay/vllm/v1/worker/gpu_model_runner.py"
frontend_serving="$root/vllm-adapter/frontend-overlay/vllm/entrypoints/openai/chat_completion/serving.py"
query_py="$root/vllm-adapter/gpu_observer_query_capture.py"
cupti_lib="$root/cupti-agent/activity/build-cuda1321/libgpu_observer_cupti_activity.so"
source_request="$root/workload/query-to-sass/foreground.json"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$root/benchmarks/lifecycle/exp0022/$stamp"
container="go-exp0022-$stamp"

mkdir -p "$run_dir/semantic-shm"
chmod 0777 "$run_dir" "$run_dir/semantic-shm"
cp "$0" "$run_dir/protocol.sh"

logs_pid=
gpu_sampler_pid=
cleanup() {
  if docker ps --format '{{.Names}}' | grep -Fxq "$container"; then
    docker stop -t 30 "$container" >/dev/null 2>&1 || true
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

for required in "$semantic_lib" "$semantic_capture" "$semantic_dump" \
  "$semantic_stats" "$full_step" "$lifecycle_client" "$lifecycle_join" \
  "$semantic_core" "$semantic_py" "$packing_runner" "$frontend_serving" \
  "$query_py" "$cupti_lib" "$source_request"; do
  [[ -f "$required" ]] || { echo "missing required artifact: $required" >&2; exit 1; }
done
docker image inspect "$image" >/dev/null

jq '.stream=true | .return_token_ids=true | .max_completion_tokens=12 |
    .ignore_eos=true | .temperature=0' \
  "$source_request" > "$run_dir/request.json"
jq '.stream=false | .max_completion_tokens=2 | .ignore_eos=true | .temperature=0' \
  "$source_request" > "$run_dir/warmup-request.json"

{
  echo "experiment=EXP-0022-request-lifecycle"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "question=Can one request be causally bracketed from client send through frontend EngineCore GPU execution and client token receipt?"
  echo "measurement_scope=diagnostic_structural_trace_not_overhead_benchmark"
  echo "model=$model"
  echo "revision=$revision"
  echo "image=$image"
  echo "external_request_id=$external_id"
  echo "execution_mode=FULL_AND_PIECEWISE_CUDA_GRAPHS"
  echo "async_scheduling=production_default"
  echo "prefix_caching=false"
  echo "semantic_abi_version=5"
  echo "engine_ring_producers=1"
  echo "frontend_ring_producers=1"
  echo "client_record_capacity=8192"
  echo "cupti_buffer_count=8"
  echo "cupti_buffer_bytes=1048576"
  echo "cupti_activity_scope=concurrent_kernels_only"
  echo "frontend_receive_endpoint=chat_handler_entry_after_HTTP_decode"
  echo "frontend_emit_endpoint=immediately_before_async_generator_yield"
  echo "client_receive_endpoint=HTTP_chunk_read_completion"
  uname -a
  free -b
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,power.draw --format=csv,noheader
  docker image inspect "$image" --format 'image_id={{.Id}} image_digests={{json .RepoDigests}}'
  sha256sum "$run_dir/request.json" "$semantic_lib" "$semantic_core" \
    "$semantic_py" "$packing_runner" "$frontend_serving" "$query_py" \
    "$cupti_lib" "$full_step" "$lifecycle_client" "$lifecycle_join"
} > "$run_dir/manifest.txt"

docker run -d \
  --name "$container" \
  --gpus all \
  --ipc=host \
  --network=host \
  --security-opt label=disable \
  -e GPU_OBSERVER_SEMANTIC_LIB=/observer/libgpu_observer_vllm_bridge.so \
  -e GPU_OBSERVER_SEMANTIC_SHM=/observer-shm/engine.ring \
  -e GPU_OBSERVER_SEMANTIC_CAPACITY=65536 \
  -e GPU_OBSERVER_FRONTEND_SHM=/observer-shm/frontend.ring \
  -e GPU_OBSERVER_FRONTEND_CAPACITY=65536 \
  -e GPU_OBSERVER_MAX_SLICES=1024 \
  -e GPU_OBSERVER_MAX_FOCUSED_TOKENS=4096 \
  -e LD_PRELOAD=/observer-cupti/libgpu_observer_cupti_activity.so \
  -e GPU_OBSERVER_CUPTI_DEFER=1 \
  -e GPU_OBSERVER_CUPTI_OUTPUT_PREFIX=/results/cupti-%p \
  -v /home/harsh4786/.cache/huggingface:/root/.cache/huggingface \
  -v "$semantic_lib:/observer/libgpu_observer_vllm_bridge.so:ro" \
  -v "$semantic_core:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py:ro" \
  -v "$semantic_py:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/gpu_observer_semantic.py:ro" \
  -v "$packing_runner:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py:ro" \
  -v "$frontend_serving:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py:ro" \
  -v "$query_py:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/gpu_observer_query_capture.py:ro" \
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
    nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw \
      --format=csv,noheader,nounits
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

docker exec "$container" chmod 0666 /observer-shm/engine.ring /observer-shm/frontend.ring
engine_pid=$(docker exec "$container" bash -lc \
  "ps -eo pid,comm | awk '\$2 ~ /^VLLM::EngineCor/ {print \$1; exit}'")
[[ "$engine_pid" =~ ^[0-9]+$ ]] || { echo "could not resolve EngineCore PID" >&2; exit 1; }
echo "$engine_pid" > "$run_dir/engine-container-pid.txt"
docker top "$container" -eo pid,comm,args > "$run_dir/process-topology.txt"
rg -F "Asynchronous scheduling is enabled" "$run_dir/server.log" \
  > "$run_dir/async-scheduling-evidence.txt" || true
rg -F "FULL_AND_PIECEWISE" "$run_dir/server.log" \
  > "$run_dir/cuda-graph-evidence.txt" || true
docker exec -e LD_PRELOAD= "$container" test -p "/tmp/gpu-observer-cupti-$engine_pid.fifo"

curl -fsS --max-time 300 \
  -H 'Content-Type: application/json' \
  -H 'X-Request-Id: exp0022-warmup' \
  --data-binary @"$run_dir/warmup-request.json" \
  "http://127.0.0.1:$port/v1/chat/completions" \
  -o "$run_dir/warmup-response.json"
"$semantic_capture" "$run_dir/semantic-shm/engine.ring" \
  "$run_dir/warmup-engine.bin" 1 > "$run_dir/warmup-engine.log" 2>&1
"$semantic_capture" "$run_dir/semantic-shm/frontend.ring" \
  "$run_dir/warmup-frontend.bin" 1 > "$run_dir/warmup-frontend.log" 2>&1

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

set +e
"$lifecycle_client" 127.0.0.1 "$port" "$external_id" \
  "$run_dir/request.json" "$run_dir/client.bin" \
  > "$run_dir/client.log" 2> "$run_dir/client.stderr"
client_status=$?
set -e
echo "$client_status" > "$run_dir/client-exit-status.txt"

"$semantic_capture" "$run_dir/semantic-shm/engine.ring" \
  "$run_dir/engine.bin" 1 > "$run_dir/engine-capture.log" 2>&1
"$semantic_capture" "$run_dir/semantic-shm/frontend.ring" \
  "$run_dir/frontend.bin" 1 > "$run_dir/frontend-capture.log" 2>&1
"$semantic_dump" "$run_dir/engine.bin" > "$run_dir/engine-dump.log"
"$semantic_dump" "$run_dir/frontend.bin" > "$run_dir/frontend-dump.log"
"$semantic_stats" "$run_dir/engine.bin" > "$run_dir/semantic-step-stats.tsv"

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

set +e
"$full_step" "$run_dir/engine.bin" "$cupti_activity" "$cupti_runtime" \
  "$cupti_summary" "$run_dir/full-step-timing.tsv" \
  "$run_dir/kernel-families.tsv" \
  > "$run_dir/full-step-analysis.log" 2> "$run_dir/full-step-analysis.stderr"
full_step_status=$?
if [[ "$full_step_status" == 0 && "$client_status" == 0 ]]; then
  "$lifecycle_join" "$run_dir/engine.bin" "$run_dir/frontend.bin" \
    "$run_dir/client.bin" "$run_dir/full-step-timing.tsv" \
    "$run_dir/lifecycle-timeline.tsv" "$run_dir/token-timeline.tsv" \
    > "$run_dir/lifecycle-analysis.log" 2> "$run_dir/lifecycle-analysis.stderr"
  lifecycle_status=$?
else
  lifecycle_status=99
fi
set -e
printf '%s\n' "$full_step_status" > "$run_dir/full-step-exit-status.txt"
printf '%s\n' "$lifecycle_status" > "$run_dir/lifecycle-exit-status.txt"

fatal_count=$(rg -i -c 'misaligned address|CUDA error|EngineCore encountered a fatal' \
  "$run_dir/server.log" || true)
fatal_count=${fatal_count:-0}
drop_count=$(rg -o 'producer_dropped=[0-9]+' \
  "$run_dir/engine-capture.log" "$run_dir/frontend-capture.log" \
  | awk -F= '{sum += $2} END {print sum + 0}')
{
  echo "server_ready=true"
  echo "client_exit_status=$client_status"
  echo "full_step_exit_status=$full_step_status"
  echo "lifecycle_exit_status=$lifecycle_status"
  echo "semantic_producer_drops=$drop_count"
  echo "fatal_count=$fatal_count"
  if [[ "$client_status" == 0 && "$full_step_status" == 0 \
    && "$lifecycle_status" == 0 && "$drop_count" == 0 && "$fatal_count" == 0 ]]; then
    echo "result=pass"
  else
    echo "result=reject"
  fi
  echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$run_dir/outcome.env"

docker logs "$container" > "$run_dir/server-final.log" 2>&1 || true
docker stop -t 30 "$container" >/dev/null
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
