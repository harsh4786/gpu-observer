#!/usr/bin/env bash
set -euo pipefail

root=/home/harsh4786/gpu-observer
image=gpu-observer/vllm-sanitizer:cuda13.2
model=Qwen/Qwen3-14B
revision=40c069824f4251a91eefaf281ebe4c544efd3e18
workload_root="$root/benchmarks/mixed-prefill/exp0012-20260811T143000Z/controlled-workload"
interactive="$workload_root/interactive-payloads-corrected.jsonl"
background="$workload_root/background-long-payloads.jsonl"
semantic_lib="$root/target/release/libgpu_observer_vllm_bridge.so"
semantic_capture="$root/target/release/semantic_capture"
semantic_stats="$root/target/release/semantic_step_stats"
join_b="$root/target/release/join_b_cache"
packing_check="$root/target/release/packing_oracle_check"
packing_runner="$root/vllm-adapter/oracle-overlay/vllm/v1/worker/gpu_model_runner.py"
packing_py="$root/vllm-adapter/oracle-overlay/vllm/v1/worker/gpu_observer_packing_oracle.py"
semantic_core="$root/vllm-adapter/overlay/vllm/v1/engine/core.py"
semantic_py="$root/vllm-adapter/gpu_observer_semantic.py"
san_build="$root/device-probes/compute-sanitizer/build"
port=8000
event_capacity=262144
run_id=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$root/benchmarks/join-b/exp0013-gate3-$run_id"
container="go-joinb-gate3-$run_id"
mkdir -p "$run_dir/semantic-shm" "$run_dir/responses"
chmod 0777 "$run_dir" "$run_dir/semantic-shm" "$run_dir/responses"

request_jobs=()
logs_pid=
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
  docker rm "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

for required in "$interactive" "$background" "$semantic_lib" "$semantic_capture" \
                "$semantic_stats" "$join_b" "$packing_check" "$packing_runner" \
                "$packing_py" "$semantic_core" "$semantic_py" \
                "$san_build/libgpu_observer_sanitizer.so" \
                "$san_build/gpu_observer_sanitizer_patches.cubin"; do
  [[ -f "$required" ]] || { echo "missing required artifact: $required" >&2; exit 1; }
done

{
  echo "experiment=EXP-0013-gate3"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "question=Does actual model-runner packed row ownership match the scheduler-derived block ownership?"
  echo "model=$model"
  echo "revision=$revision"
  echo "image=$image"
  echo "execution_mode=eager"
  echo "async_scheduling=false"
  echo "prefix_caching=false"
  echo "target_kernel=reshape_and_cache_flash_kernel"
  echo "event_capacity=$event_capacity"
  echo "event_size_bytes=64"
  echo "event_buffer_bytes=$((event_capacity * 64))"
  echo "interactive_requests=8"
  echo "interactive_output_tokens=16"
  echo "injection_delay_seconds=0.5"
  echo "ownership=packing_oracle_required"
  uname -a
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,power.draw --format=csv,noheader
  docker image inspect "$image" --format 'image_id={{.Id}} image_digests={{json .RepoDigests}}'
  sha256sum "$interactive" "$background" "$semantic_lib" "$semantic_core" "$semantic_py" \
            "$packing_runner" "$packing_py" "$packing_check" \
            "$san_build/libgpu_observer_sanitizer.so" \
            "$san_build/gpu_observer_sanitizer_patches.cubin" "$join_b"
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
  -e GPU_OBSERVER_PACKING_ORACLE=/results/packing-oracle.bin \
  -e GPU_OBSERVER_PACKING_MAX_REQUESTS=1024 \
  -e GPU_OBSERVER_SAN_EVENT_CAPACITY="$event_capacity" \
  -e GPU_OBSERVER_SAN_KERNEL_SUBSTRING=reshape_and_cache_flash_kernel \
  -e GPU_OBSERVER_SAN_LAUNCH_ID=1 \
  -v /home/harsh4786/.cache/huggingface:/root/.cache/huggingface \
  -v "$semantic_lib:/observer/libgpu_observer_vllm_bridge.so:ro" \
  -v "$semantic_core:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py:ro" \
  -v "$semantic_py:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/gpu_observer_semantic.py:ro" \
  -v "$packing_runner:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py:ro" \
  -v "$packing_py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_observer_packing_oracle.py:ro" \
  -v "$san_build:/observer-sanitizer:ro" \
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
  --enforce-eager \
  --no-async-scheduling \
  --no-enable-prefix-caching > "$run_dir/container-id.txt"
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
docker exec "$container" chmod 0666 /observer-shm/semantic.ring
engine_pid=$(docker exec "$container" bash -lc \
  "ps -eo pid,comm | awk '\$2 ~ /^VLLM::EngineCor/ {print \$1; exit}'")
[[ -n "$engine_pid" ]] || { echo "could not resolve EngineCore PID" >&2; exit 1; }
echo "$engine_pid" > "$run_dir/engine-container-pid.txt"
docker top "$container" -eo pid,comm,args > "$run_dir/process-topology.txt"

head -n 1 "$interactive" | jq '.max_completion_tokens = 1' | curl -fsS \
  -H 'Content-Type: application/json' \
  -H 'X-Request-Id: exp0013-warmup' \
  --data-binary @- \
  "http://127.0.0.1:$port/v1/chat/completions" \
  -o "$run_dir/warmup-response.json"
"$semantic_capture" "$run_dir/semantic-shm/semantic.ring" \
  "$run_dir/warmup-semantic.bin" 1 > "$run_dir/warmup-semantic.log" 2>&1

# First SIGUSR2 resets device counters and launch IDs on the next CUDA launch.
docker exec "$container" kill -USR2 "$engine_pid"
echo "capture_armed_after_warmup=true" >> "$run_dir/manifest.txt"

for index in $(seq 1 8); do
  sed -n "${index}p" "$interactive" | jq '.max_completion_tokens = 16' | curl -fsS -N \
    -H 'Content-Type: application/json' \
    -H "X-Request-Id: exp0013-short-$index" \
    --data-binary @- \
    "http://127.0.0.1:$port/v1/chat/completions" \
    > "$run_dir/responses/short-$index.sse" \
    2> "$run_dir/responses/short-$index.stderr" &
  request_jobs+=("$!")
done
sleep 0.5
sed -n '1p' "$background" | curl -fsS -N \
  -H 'Content-Type: application/json' \
  -H 'X-Request-Id: exp0013-long-1' \
  --data-binary @- \
  "http://127.0.0.1:$port/v1/chat/completions" \
  > "$run_dir/responses/long-1.sse" \
  2> "$run_dir/responses/long-1.stderr" &
request_jobs+=("$!")

request_status=0
for job in "${request_jobs[@]}"; do
  wait "$job" || request_status=1
done
request_jobs=()
echo "$request_status" > "$run_dir/request-exit-status.txt"
[[ "$request_status" == 0 ]] || { echo "one or more requests failed" >&2; exit 1; }

"$semantic_capture" "$run_dir/semantic-shm/semantic.ring" \
  "$run_dir/semantic.bin" 1 > "$run_dir/semantic-capture.log" 2>&1
"$semantic_stats" "$run_dir/semantic.bin" > "$run_dir/semantic-step-stats.tsv"

# Second SIGUSR2 requests a snapshot; the tiny request only triggers the host callback.
docker exec "$container" kill -USR2 "$engine_pid"
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
  "$run_dir/device-probe.events.bin" > "$run_dir/join-b-cache.log"
rg '^step=.*prefill=[1-9][0-9]*.*decode=[1-9][0-9]*' "$run_dir/join-b-cache.log" \
  > "$run_dir/mixed-steps.txt"
[[ -s "$run_dir/mixed-steps.txt" ]] || { echo "workload produced no mixed step" >&2; exit 1; }

docker logs "$container" > "$run_dir/server-final.log" 2>&1 || true
docker exec "$container" chmod 0644 /results/packing-oracle.bin
docker stop --time 30 "$container" >/dev/null
wait "$logs_pid" 2>/dev/null || true
logs_pid=
docker rm "$container" >/dev/null
{
  echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader
} > "$run_dir/final-state.txt"
packing_status=0
"$packing_check" "$run_dir/semantic.bin" "$run_dir/packing-oracle.bin" \
  > "$run_dir/packing-oracle.log" 2>&1 || packing_status=$?
echo "$packing_status" > "$run_dir/packing-oracle-exit-status.txt"
chmod -R a+rX "$run_dir" 2>/dev/null || true
"$root/benchmarks/seal-run.sh" "$run_dir"
trap - EXIT INT TERM
if [[ "$packing_status" -ne 0 ]]; then
  echo "Gate 3 rejected scheduler-derived request ownership; see $run_dir/packing-oracle.log" >&2
  exit "$packing_status"
fi
printf 'completed Gate 3: %s\n' "$run_dir"
