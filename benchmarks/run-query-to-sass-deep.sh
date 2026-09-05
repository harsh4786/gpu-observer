#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 TIMED_RUN_DIR" >&2
  exit 2
fi

root=/home/harsh4786/gpu-observer
timed_run=$(readlink -f "$1")
[[ -d "$timed_run" ]] || { echo "timed run not found: $timed_run" >&2; exit 1; }
[[ -s "$timed_run/run.json" && -s "$timed_run/query.msgpack.frames" \
  && -s "$timed_run/semantic.bin" ]] || { echo "timed run is incomplete" >&2; exit 1; }

image=$(jq -r '.image' "$timed_run/run.json")
model=$(jq -r '.model' "$timed_run/run.json")
revision=$(jq -r '.modelRevision' "$timed_run/run.json")
expected_signature=$(jq -r '.semanticSignature' "$timed_run/run.json")
target_kernel=reshape_and_cache_flash_kernel
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$root/benchmarks/query-to-sass/$stamp-deep"
container="go-query-sass-deep-$stamp"
probe_build="$root/device-probes/compute-sanitizer/build"
semantic_lib="$root/target/release/libgpu_observer_vllm_bridge.so"
semantic_capture="$root/target/release/semantic_capture"
semantic_dump="$root/target/release/semantic_dump"
deep_exporter="$root/target/release/export_deep_replay"
trace_exporter="$root/target/release/export_trace_bundle"
semantic_core="$root/vllm-adapter/overlay/vllm/v1/engine/core.py"
semantic_py="$root/vllm-adapter/gpu_observer_semantic.py"
attention_observer="$root/vllm-adapter/gpu_observer_attention.py"
packing_runner="$root/vllm-adapter/packed-overlay/vllm/v1/worker/gpu_model_runner.py"
port=8000

cd "$root"
cargo build --release -p gpu-observer-collector --bins -p gpu-observer-vllm-bridge
docker run --rm \
  -v "$root/device-probes/compute-sanitizer:/src" \
  -w /src \
  nvcr.io/nvidia/cuda:13.2.1-devel-ubuntu24.04 \
  make CUDA_PATH=/usr/local/cuda

mkdir -p "$run_dir/semantic-shm" "$run_dir/responses"
chmod 0777 "$run_dir" "$run_dir/semantic-shm" "$run_dir/responses"

request_jobs=()
logs_pid=
cleanup() {
  for job in "${request_jobs[@]}"; do kill "$job" 2>/dev/null || true; done
  if docker ps --format '{{.Names}}' | grep -Fxq "$container"; then
    docker stop --time 30 "$container" >/dev/null 2>&1 || true
  fi
  if [[ -n "$logs_pid" ]]; then wait "$logs_pid" 2>/dev/null || true; fi
  docker rm "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

{
  echo "experiment=QUERY-TO-SASS-DEEP"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "timed_run=$timed_run"
  echo "model=$model"
  echo "revision=$revision"
  echo "image=$image"
  echo "target_kernel=$target_kernel"
  echo "mode=block_event"
  echo "callback_scope=function"
  echo "capture=signal_armed_after_warmup"
  echo "expected_semantic_signature=$expected_signature"
  sha256sum "$probe_build/libgpu_observer_sanitizer.so" \
    "$probe_build/gpu_observer_sanitizer_patches.cubin" \
    "$timed_run/requests/focus.json" "$semantic_lib" "$deep_exporter"
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
  -e GPU_OBSERVER_FOCUS_REQUEST_ID=chatcmpl-go-agentic-debug-foreground \
  -e LD_PRELOAD=/observer-sanitizer/libgpu_observer_sanitizer.so \
  -e GPU_OBSERVER_SAN_MODE=block_event \
  -e GPU_OBSERVER_SAN_PATCH_FILE=/observer-sanitizer/gpu_observer_sanitizer_patches.cubin \
  -e GPU_OBSERVER_SAN_OUTPUT_PREFIX=/results/device-probe \
  -e GPU_OBSERVER_SAN_EVENT_CAPACITY=1048576 \
  -e GPU_OBSERVER_SAN_KERNEL_SUBSTRING="$target_kernel" \
  -e GPU_OBSERVER_SAN_CALLBACK_DATA_SCOPE=function \
  -e GPU_OBSERVER_SAN_GRAPH_NODES=1 \
  -e GPU_OBSERVER_SAN_PASSIVE_LAUNCHES=1 \
  -e GPU_OBSERVER_SAN_ARM_ON_SIGNAL=1 \
  -v /home/harsh4786/.cache/huggingface:/root/.cache/huggingface \
  -v "$semantic_lib:/observer/libgpu_observer_vllm_bridge.so:ro" \
  -v "$semantic_core:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py:ro" \
  -v "$semantic_py:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/gpu_observer_semantic.py:ro" \
  -v "$attention_observer:/usr/local/lib/python3.12/dist-packages/vllm/gpu_observer_attention.py:ro" \
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
  if ! docker ps --format '{{.Names}}' | grep -Fxq "$container"; then break; fi
  sleep 1
done
[[ "$ready" == 1 ]] || { docker logs "$container"; exit 1; }

docker exec "$container" chmod 0666 /observer-shm/semantic.ring
engine_pid=$(docker exec "$container" bash -lc \
  "ps -eo pid,comm | awk '\$2 ~ /^VLLM::EngineCor/ {print \$1; exit}'")
[[ "$engine_pid" =~ ^[0-9]+$ ]] || { echo "could not resolve EngineCore PID" >&2; exit 1; }
echo "$engine_pid" > "$run_dir/engine-container-pid.txt"

curl -fsS --max-time 300 \
  -H 'Content-Type: application/json' \
  -H 'X-Request-Id: query-to-sass-warmup' \
  --data-binary @"$timed_run/requests/warmup.json" \
  "http://127.0.0.1:$port/v1/chat/completions" \
  -o "$run_dir/responses/warmup.json"
"$semantic_capture" "$run_dir/semantic-shm/semantic.ring" \
  "$run_dir/warmup-semantic.bin" 1 > "$run_dir/warmup-semantic.log" 2>&1

docker exec "$container" kill -USR2 "$engine_pid"
run_request() {
  local request_id=$1
  local request_path=$2
  local response_path=$3
  curl -fsS --max-time 900 \
    -H 'Content-Type: application/json' \
    -H "X-Request-Id: $request_id" \
    --data-binary @"$request_path" \
    "http://127.0.0.1:$port/v1/chat/completions" \
    -o "$response_path"
}

for index in $(seq 1 7); do
  run_request "query-to-sass-background-$index" \
    "$timed_run/requests/background-$index.json" \
    "$run_dir/responses/background-$index.json" &
  request_jobs+=("$!")
done
sleep 0.15
run_request go-agentic-debug-foreground \
  "$timed_run/requests/focus.json" \
  "$run_dir/responses/focus.json" &
request_jobs+=("$!")

request_status=0
for job in "${request_jobs[@]}"; do wait "$job" || request_status=1; done
request_jobs=()
[[ "$request_status" == 0 ]] || { echo "deep workload request failed" >&2; exit 1; }

"$semantic_capture" "$run_dir/semantic-shm/semantic.ring" \
  "$run_dir/semantic.bin" 1 > "$run_dir/semantic-capture.log" 2>&1
"$semantic_dump" "$run_dir/semantic.bin" > "$run_dir/semantic-dump.log"
sed -E 's/ts=[0-9]+ //g; s/seq=[0-9]+ //g' \
  "$run_dir/semantic-dump.log" > "$run_dir/semantic-canonical.log"
semantic_signature=$(sha256sum "$run_dir/semantic-canonical.log" | awk '{print $1}')
echo "observed_semantic_signature=$semantic_signature" >> "$run_dir/manifest.txt"
[[ "$semantic_signature" == "$expected_signature" ]] || {
  echo "matched replay rejected: semantic signatures differ" >&2
  exit 1
}

docker cp \
  "$container:/usr/local/lib/python3.12/dist-packages/vllm/_C.abi3.so" \
  "$run_dir/vllm_C.abi3.so"
module_hash=$(sha256sum "$run_dir/vllm_C.abi3.so" | awk '{print $1}')

docker exec "$container" kill -USR2 "$engine_pid"
curl -fsS --max-time 300 \
  -H 'Content-Type: application/json' \
  -H 'X-Request-Id: query-to-sass-flush' \
  --data-binary @"$timed_run/requests/warmup.json" \
  "http://127.0.0.1:$port/v1/chat/completions" \
  -o "$run_dir/responses/flush.json"
for _ in $(seq 1 200); do
  [[ -s "$run_dir/device-probe.summary.tsv" \
    && -s "$run_dir/device-probe.events.bin" ]] && break
  sleep 0.1
done
[[ -s "$run_dir/device-probe.summary.tsv" \
  && -s "$run_dir/device-probe.events.bin" ]] || { echo "sanitizer flush failed" >&2; exit 1; }

cuobjdump --list-text "$run_dir/vllm_C.abi3.so" \
  > "$run_dir/vllm_C.text-sections.txt"
function_index=$(awk '
  /reshape_and_cache_flash_kernelI13__nv_bfloat16S1_/ && /\.sm_120\.elf\.bin$/ {
    print $4
    exit
  }' "$run_dir/vllm_C.text-sections.txt")
[[ "$function_index" =~ ^[0-9]+$ ]] || { echo "BF16 sm_120 cache-kernel SASS not found" >&2; exit 1; }
cuobjdump --dump-sass --function-index "$function_index" \
  "$run_dir/vllm_C.abi3.so" > "$run_dir/reshape-and-cache.sass"

"$deep_exporter" \
  --run "$timed_run/run.json" \
  --summary "$run_dir/device-probe.summary.tsv" \
  --events "$run_dir/device-probe.events.bin" \
  --sass "$run_dir/reshape-and-cache.sass" \
  --module-hash "$module_hash" \
  --target "$target_kernel" \
  --output "$run_dir/deep-replay.json" \
  > "$run_dir/deep-export.log"

mapfile -t activities < <(find "$timed_run" -maxdepth 1 -name 'cupti-*.activities.tsv' -type f)
mapfile -t runtimes < <(find "$timed_run" -maxdepth 1 -name 'cupti-*.runtime.tsv' -type f)
mapfile -t summaries < <(find "$timed_run" -maxdepth 1 -name 'cupti-*.summary.tsv' -type f)
[[ "${#activities[@]}" == 1 && "${#runtimes[@]}" == 1 && "${#summaries[@]}" == 1 ]] || {
  echo "timed run CUPTI artifact selection is ambiguous" >&2
  exit 1
}

"$trace_exporter" \
  --query "$timed_run/query.msgpack.frames" \
  --semantic "$timed_run/semantic.bin" \
  --activities "${activities[0]}" \
  --runtime "${runtimes[0]}" \
  --summary "${summaries[0]}" \
  --run "$timed_run/run.json" \
  --deep "$run_dir/deep-replay.json" \
  --output "$run_dir/trace-bundle-v2-with-sass.json" \
  > "$run_dir/trace-export.log"

fatal_count=$(rg -i -c 'misaligned address|CUDA error|EngineCore encountered a fatal' \
  "$run_dir/server.log" || true)
fatal_count=${fatal_count:-0}
{
  echo "server_ready=true"
  echo "request_exit_status=$request_status"
  echo "semantic_signature_match=true"
  echo "module_hash=$module_hash"
  echo "function_index=$function_index"
  echo "fatal_count=$fatal_count"
  if [[ "$fatal_count" == 0 && -s "$run_dir/trace-bundle-v2-with-sass.json" ]]; then
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
docker rm "$container" >/dev/null
"$root/benchmarks/seal-run.sh" "$run_dir"
trap - EXIT INT TERM
printf '%s\n' "$run_dir"
printf 'visualize: python3 -m http.server 8088 --directory %s\n' "$root"
printf 'open: http://127.0.0.1:8088/ui/trace.html?trace=../benchmarks/query-to-sass/%s-deep/trace-bundle-v2-with-sass.json\n' "$stamp"

