#!/usr/bin/env bash
set -euo pipefail

root=/home/harsh4786/gpu-observer
image=gpu-observer/vllm-sanitizer:cuda13.2
model=Qwen/Qwen3-14B
revision=40c069824f4251a91eefaf281ebe4c544efd3e18
backgrounds="$root/benchmarks/mixed-prefill/exp0012-20260811T143000Z/controlled-workload/interactive-payloads-corrected.jsonl"
focus_payload="${GPU_OBSERVER_FOCUS_PAYLOAD:-$root/workload/query-to-sass/foreground.json}"
semantic_lib="$root/target/release/libgpu_observer_vllm_bridge.so"
semantic_capture="$root/target/release/semantic_capture"
semantic_dump="$root/target/release/semantic_dump"
query_capture="$root/target/release/query_capture"
trace_exporter="$root/target/release/export_trace_bundle"
semantic_core="$root/vllm-adapter/overlay/vllm/v1/engine/core.py"
semantic_py="$root/vllm-adapter/gpu_observer_semantic.py"
attention_observer="$root/vllm-adapter/gpu_observer_attention.py"
packing_runner="$root/vllm-adapter/packed-overlay/vllm/v1/worker/gpu_model_runner.py"
frontend_serving="$root/vllm-adapter/frontend-overlay/vllm/entrypoints/openai/chat_completion/serving.py"
query_py="$root/vllm-adapter/gpu_observer_query_capture.py"
cupti_lib="$root/cupti-agent/activity/build-cuda1321/libgpu_observer_cupti_activity.so"
focus_external=go-agentic-debug-foreground
focus_internal=chatcmpl-go-agentic-debug-foreground
port=8000
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$root/benchmarks/query-to-sass/$stamp-timed${GPU_OBSERVER_RUN_SUFFIX:-}"
container="go-query-sass-$stamp"

cd "$root"
cargo build --release -p gpu-observer-collector --bins -p gpu-observer-vllm-bridge

mkdir -p "$run_dir/semantic-shm" "$run_dir/requests" "$run_dir/responses"
chmod 0777 "$run_dir" "$run_dir/semantic-shm" "$run_dir/requests" "$run_dir/responses"

for required in "$backgrounds" "$focus_payload" "$semantic_lib" "$semantic_capture" \
  "$semantic_dump" "$query_capture" "$trace_exporter" "$semantic_core" \
  "$semantic_py" "$attention_observer" "$packing_runner" "$frontend_serving" "$query_py" "$cupti_lib"; do
  [[ -f "$required" ]] || { echo "missing required artifact: $required" >&2; exit 1; }
done
docker image inspect "$image" >/dev/null

request_jobs=()
logs_pid=
gpu_sampler_pid=
query_capture_pid=
cleanup() {
  for job in "${request_jobs[@]}"; do kill "$job" 2>/dev/null || true; done
  if [[ -n "$query_capture_pid" ]]; then kill "$query_capture_pid" 2>/dev/null || true; fi
  if docker ps --format '{{.Names}}' | grep -Fxq "$container"; then
    docker stop --time 30 "$container" >/dev/null 2>&1 || true
  fi
  if [[ -n "$logs_pid" ]]; then wait "$logs_pid" 2>/dev/null || true; fi
  if [[ -n "$gpu_sampler_pid" ]]; then
    kill "$gpu_sampler_pid" 2>/dev/null || true
    wait "$gpu_sampler_pid" 2>/dev/null || true
  fi
  docker rm "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

cp "$focus_payload" "$run_dir/requests/focus.json"
{
  sha256sum "$run_dir/requests/focus.json"
  printf '%s\n' "$model" "$revision" "$image"
  printf '%s\n' \
    '--dtype=bfloat16' '--max-model-len=4096' '--kv-cache-memory-bytes=8G' \
    '--max-num-seqs=32' '--no-async-scheduling' '--no-enable-prefix-caching'
} > "$run_dir/replay-fingerprint-material.txt"
replay_fingerprint=$(sha256sum "$run_dir/replay-fingerprint-material.txt" | awk '{print $1}')

jq -n \
  --arg experiment "QUERY-TO-SASS-TIMED" \
  --arg started "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg model "$model" \
  --arg revision "$revision" \
  --arg image "$image" \
  --arg fingerprint "$replay_fingerprint" \
  '{
    experiment: $experiment,
    startedUtc: $started,
    captureStatus: "measured_sealed_trace",
    model: $model,
    modelRevision: $revision,
    image: $image,
    executionMode: "CUDA Graph, synchronous scheduling",
    replayFingerprint: $fingerprint,
    focusPolicy: "one explicit foreground request",
    cuptiFilter: "reshape_and_cache_flash_kernel",
    semanticAbi: 3,
    notes: [
      "CUPTI is the actual-time source.",
      "Exact block ownership is claimed only for the validated cache kernel.",
      "A deep sanitizer replay must use the same replayFingerprint."
    ]
  }' > "$run_dir/run.json"

{
  echo "experiment=QUERY-TO-SASS-TIMED"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "model=$model"
  echo "revision=$revision"
  echo "image=$image"
  echo "focus_external=$focus_external"
  echo "focus_internal=$focus_internal"
  echo "replay_fingerprint=$replay_fingerprint"
  echo "scheduling=synchronous"
  echo "execution=CUDA_Graph"
  echo "cupti_filter=reshape_and_cache_flash_kernel"
  uname -a
  free -b
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,power.draw --format=csv,noheader
  docker image inspect "$image" --format 'image_id={{.Id}} image_digests={{json .RepoDigests}}'
  sha256sum "$focus_payload" "$semantic_lib" "$semantic_core" "$semantic_py" \
    "$packing_runner" "$frontend_serving" "$query_py" "$cupti_lib" "$trace_exporter"
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
  -e GPU_OBSERVER_MAX_FOCUSED_TOKENS=8192 \
  -e GPU_OBSERVER_FOCUS_REQUEST_ID="$focus_internal" \
  -e GPU_OBSERVER_FOCUS_EXTERNAL_ID="$focus_external" \
  -e GPU_OBSERVER_QUERY_SOCKET=/observer-shm/query.sock \
  -e GPU_OBSERVER_QUERY_MAX_BYTES=49152 \
  -e LD_PRELOAD=/observer-cupti/libgpu_observer_cupti_activity.so \
  -e GPU_OBSERVER_CUPTI_DEFER=1 \
  -e GPU_OBSERVER_CUPTI_OUTPUT_PREFIX=/results/cupti-%p \
  -e GPU_OBSERVER_CUPTI_KERNEL_SUBSTRING=reshape_and_cache_flash_kernel \
  -v /home/harsh4786/.cache/huggingface:/root/.cache/huggingface \
  -v "$semantic_lib:/observer/libgpu_observer_vllm_bridge.so:ro" \
  -v "$semantic_core:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py:ro" \
  -v "$semantic_py:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/gpu_observer_semantic.py:ro" \
  -v "$attention_observer:/usr/local/lib/python3.12/dist-packages/vllm/gpu_observer_attention.py:ro" \
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
  --no-async-scheduling \
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
  if ! docker ps --format '{{.Names}}' | grep -Fxq "$container"; then break; fi
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

jq '.request_id="query-to-sass-warmup" | .max_completion_tokens=2' \
  "$focus_payload" > "$run_dir/requests/warmup.json"
curl -fsS --max-time 300 \
  -H 'Content-Type: application/json' \
  -H 'X-Request-Id: query-to-sass-warmup' \
  --data-binary @"$run_dir/requests/warmup.json" \
  "http://127.0.0.1:$port/v1/chat/completions" \
  -o "$run_dir/responses/warmup.json"
"$semantic_capture" "$run_dir/semantic-shm/semantic.ring" \
  "$run_dir/warmup-semantic.bin" 1 > "$run_dir/warmup-semantic.log" 2>&1

"$query_capture" "$run_dir/semantic-shm/query.sock" \
  "$run_dir/query.msgpack.frames" 2 900000 > "$run_dir/query-capture.log" 2>&1 &
query_capture_pid=$!
for _ in $(seq 1 100); do
  [[ -S "$run_dir/semantic-shm/query.sock" ]] && break
  sleep 0.05
done
[[ -S "$run_dir/semantic-shm/query.sock" ]] || { echo "query socket did not appear" >&2; exit 1; }
chmod 0666 "$run_dir/semantic-shm/query.sock"

docker exec -e LD_PRELOAD= "$container" sh -c \
  "printf S > /tmp/gpu-observer-cupti-$engine_pid.fifo"
for _ in $(seq 1 200); do
  rg -Fq "gpu-observer-cupti: control command=S status=0" "$run_dir/server.log" && break
  sleep 0.1
done
rg -Fq "gpu-observer-cupti: control command=S status=0" "$run_dir/server.log"

run_request() {
  local request_id=$1
  local request_path=$2
  local response_path=$3
  local metrics_path=$4
  curl -sS --max-time 900 \
    -H 'Content-Type: application/json' \
    -H "X-Request-Id: $request_id" \
    --data-binary @"$request_path" \
    -o "$response_path" \
    -w 'http_code=%{http_code}\ntime_starttransfer_s=%{time_starttransfer}\ntime_total_s=%{time_total}\n' \
    "http://127.0.0.1:$port/v1/chat/completions" > "$metrics_path"
}

for index in $(seq 1 7); do
  sed -n "${index}p" "$backgrounds" | jq \
    --arg request_id "query-to-sass-background-$index" \
    '.request_id=$request_id | .stream=false | .max_completion_tokens=64 | .ignore_eos=true | .temperature=0 | .seed=7' \
    > "$run_dir/requests/background-$index.json"
  run_request "query-to-sass-background-$index" \
    "$run_dir/requests/background-$index.json" \
    "$run_dir/responses/background-$index.json" \
    "$run_dir/responses/background-$index.metrics" &
  request_jobs+=("$!")
done

sleep 0.15
run_request "$focus_external" "$run_dir/requests/focus.json" \
  "$run_dir/responses/focus.json" "$run_dir/responses/focus.metrics" &
focus_job=$!
request_jobs+=("$focus_job")

request_status=0
for job in "${request_jobs[@]}"; do wait "$job" || request_status=1; done
request_jobs=()
wait "$query_capture_pid" || true
query_capture_pid=
rg -Fq "query_capture frames=2 " "$run_dir/query-capture.log"

"$semantic_capture" "$run_dir/semantic-shm/semantic.ring" \
  "$run_dir/semantic.bin" 1 > "$run_dir/semantic-capture.log" 2>&1
"$semantic_dump" "$run_dir/semantic.bin" > "$run_dir/semantic-dump.log"
sed -E 's/ts=[0-9]+ //g; s/seq=[0-9]+ //g' \
  "$run_dir/semantic-dump.log" > "$run_dir/semantic-canonical.log"
semantic_signature=$(sha256sum "$run_dir/semantic-canonical.log" | awk '{print $1}')
jq --arg signature "$semantic_signature" '.semanticSignature=$signature' \
  "$run_dir/run.json" > "$run_dir/run.next.json"
mv "$run_dir/run.next.json" "$run_dir/run.json"
echo "semantic_signature=$semantic_signature" >> "$run_dir/manifest.txt"

docker exec -e LD_PRELOAD= "$container" sh -c \
  "printf F > /tmp/gpu-observer-cupti-$engine_pid.fifo"
cupti_activity="$run_dir/cupti-$engine_pid.activities.tsv"
cupti_runtime="$run_dir/cupti-$engine_pid.runtime.tsv"
cupti_summary="$run_dir/cupti-$engine_pid.summary.tsv"
for _ in $(seq 1 300); do
  if [[ -s "$cupti_summary" && -s "$cupti_activity" && -s "$cupti_runtime" ]] \
    && rg -Fq "gpu-observer-cupti: control command=F status=0" "$run_dir/server.log"; then
    break
  fi
  sleep 0.1
done
for required in "$cupti_summary" "$cupti_activity" "$cupti_runtime"; do
  [[ -s "$required" ]] || { echo "missing CUPTI output: $required" >&2; exit 1; }
done

"$trace_exporter" \
  --query "$run_dir/query.msgpack.frames" \
  --semantic "$run_dir/semantic.bin" \
  --activities "$cupti_activity" \
  --runtime "$cupti_runtime" \
  --summary "$cupti_summary" \
  --run "$run_dir/run.json" \
  --output "$run_dir/trace-bundle-v2.json" \
  > "$run_dir/export.log"

focus_http=$(awk -F= '$1=="http_code" {print $2}' "$run_dir/responses/focus.metrics")
focus_tokens=$(jq -r '.usage.completion_tokens // 0' "$run_dir/responses/focus.json")
fatal_count=$(rg -i -c 'misaligned address|CUDA error|EngineCore encountered a fatal' \
  "$run_dir/server.log" || true)
fatal_count=${fatal_count:-0}
{
  echo "server_ready=true"
  echo "request_exit_status=$request_status"
  echo "query_frames=2"
  echo "focus_http=$focus_http"
  echo "focus_completion_tokens=$focus_tokens"
  echo "fatal_count=$fatal_count"
  if [[ "$request_status" == 0 && "$focus_http" == 200 && "$focus_tokens" -gt 0 \
    && "$fatal_count" == 0 && -s "$run_dir/trace-bundle-v2.json" ]]; then
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
kill "$gpu_sampler_pid" 2>/dev/null || true
wait "$gpu_sampler_pid" 2>/dev/null || true
gpu_sampler_pid=
docker rm "$container" >/dev/null
"$root/benchmarks/seal-run.sh" "$run_dir"
trap - EXIT INT TERM
printf '%s\n' "$run_dir"
printf 'visualize: python3 -m http.server 8088 --directory %s\n' "$root"
printf 'open: http://127.0.0.1:8088/ui/trace.html?trace=../benchmarks/query-to-sass/%s-timed/trace-bundle-v2.json\n' "$stamp"

