#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/env.sh"
source "$script_dir/active-run.env"
cd "$GO_ROOT"

attempt_id=$(date -u +%Y%m%dT%H%M%SZ)
attempt_dir="$GO_RUN/cupti-nsys-$attempt_id"
semantic_dir="$attempt_dir/semantic-shm"
cupti_container="gpu-observer-cupti-q14-$attempt_id"
session_name="Q14C${attempt_id}"
mkdir -p "$semantic_dir"

semantic_job=""
cleanup() {
    if [[ -n "$semantic_job" ]] && kill -0 "$semantic_job" 2>/dev/null; then
        kill "$semantic_job" 2>/dev/null || true
    fi
    docker stop --time 30 "$cupti_container" >/dev/null 2>&1 || true
    docker rm -f "$cupti_container" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

clock_pair() {
    python3 -c 'import time; m0=time.monotonic_ns(); r=time.time_ns(); m1=time.monotonic_ns(); print(f"monotonic_ns={(m0+m1)//2} realtime_ns={r} uncertainty_ns={m1-m0}")'
}

echo "[1/8] Stop the ordinary server; CUPTI must own the process from CUDA initialization"
docker stop "$GO_CONTAINER" >/dev/null 2>&1 || true

echo "[2/8] Start Qwen3-14B under Nsight/CUPTI with collection paused"
docker run -d \
    --name "$cupti_container" \
    --gpus all \
    --ipc=host \
    --network=host \
    --ulimit memlock=-1 \
    --security-opt label=disable \
    -e GPU_OBSERVER_SEMANTIC_LIB=/observer/libgpu_observer_vllm_bridge.so \
    -e GPU_OBSERVER_SEMANTIC_SHM=/observer-shm/semantic.ring \
    -e GPU_OBSERVER_SEMANTIC_CAPACITY=65536 \
    -e GPU_OBSERVER_MAX_SLICES=1024 \
    -v "$GO_NSYS_ROOT:$GO_NSYS_ROOT:ro" \
    -v "$GO_HF_CACHE:/root/.cache/huggingface" \
    -v "$GO_SEMANTIC_LIB:/observer/libgpu_observer_vllm_bridge.so:ro" \
    -v "$GO_SEMANTIC_CORE:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py:ro" \
    -v "$GO_SEMANTIC_PY:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/gpu_observer_semantic.py:ro" \
    -v "$semantic_dir:/observer-shm" \
    -v "$attempt_dir:/results" \
    --entrypoint "$GO_NSYS_BIN" \
    "$GO_IMAGE" \
    profile \
    --start-later=true \
    --session-new="$session_name" \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --cuda-graph-trace=node \
    --cuda-flush-interval=1000 \
    --export=sqlite \
    --force-overwrite=true \
    --output=/results/nsys-qwen14-eager \
    vllm serve "$GO_MODEL" \
    --revision "$GO_MODEL_REVISION" \
    --port "$GO_PORT" \
    --dtype bfloat16 \
    --enforce-eager \
    --no-async-scheduling \
    --no-enable-prefix-caching \
    --max-model-len 4096 \
    --kv-cache-memory-bytes 8G \
    --max-num-seqs 32 >"$attempt_dir/container-id.txt"

echo "[3/8] Wait for model load while CUPTI collection remains paused"
server_ready=0
for _ in $(seq 1 240); do
    if curl -fsS -o /dev/null "http://127.0.0.1:$GO_PORT/health"; then
        server_ready=1
        break
    fi
    if ! docker ps --format '{{.Names}}' | rg -qx "$cupti_container"; then
        echo "error: profiled vLLM container exited during startup" >&2
        docker logs "$cupti_container" >&2 || true
        exit 1
    fi
    sleep 2
done
if [[ "$server_ready" -ne 1 ]]; then
    echo "error: Qwen3-14B was not healthy after eight minutes" >&2
    docker logs --tail 160 "$cupti_container" >&2 || true
    exit 1
fi
docker logs "$cupti_container" >"$attempt_dir/server-ready.log" 2>&1
docker exec "$cupti_container" chmod 0666 /observer-shm/semantic.ring

echo "[4/8] Warm the exact request and drain its semantic records"
curl -fsS \
    -H 'Content-Type: application/json' \
    --data-binary "@$script_dir/request.json" \
    "http://127.0.0.1:$GO_PORT/v1/chat/completions" \
    -o "$attempt_dir/warmup-response.json"
"$GO_SEMANTIC_CAPTURE" \
    "$semantic_dir/semantic.ring" \
    "$attempt_dir/warmup-semantic.bin" \
    1 >"$attempt_dir/warmup-semantic.log" 2>&1

echo "[5/8] Start semantic capture, then start the delayed CUPTI session"
"$GO_SEMANTIC_CAPTURE" \
    "$semantic_dir/semantic.ring" \
    "$attempt_dir/semantic.bin" \
    20 >"$attempt_dir/semantic-capture.log" 2>&1 &
semantic_job=$!
sleep 0.2
if ! kill -0 "$semantic_job" 2>/dev/null; then
    echo "error: semantic capture failed to start" >&2
    sed -n '1,120p' "$attempt_dir/semantic-capture.log" >&2
    exit 1
fi
docker exec "$cupti_container" "$GO_NSYS_BIN" start \
    --session="$session_name" >"$attempt_dir/nsys-start.log" 2>&1

echo "[6/8] Record clock pair, send one request, and record the second clock pair"
clock_pair >"$attempt_dir/client-clock-before.txt"
curl -fsS \
    -H 'Content-Type: application/json' \
    --data-binary "@$script_dir/request.json" \
    "http://127.0.0.1:$GO_PORT/v1/chat/completions" \
    -o "$attempt_dir/response.json"
clock_pair >"$attempt_dir/client-clock-after.txt"

echo "[7/8] Stop CUPTI immediately and export the raw report plus SQLite"
docker exec "$cupti_container" "$GO_NSYS_BIN" stop \
    --session="$session_name" >"$attempt_dir/nsys-stop.log" 2>&1
wait "$semantic_job"
semantic_job=""
docker logs "$cupti_container" >"$attempt_dir/server-full.log" 2>&1 || true

report="$attempt_dir/nsys-qwen14-eager.nsys-rep"
database="$attempt_dir/nsys-qwen14-eager.sqlite"
if [[ ! -s "$report" || ! -s "$database" ]]; then
    echo "error: Nsight did not produce both non-empty report and SQLite files" >&2
    sed -n '1,200p' "$attempt_dir/nsys-stop.log" >&2
    exit 1
fi
if [[ ! -s "$attempt_dir/semantic.bin" ]]; then
    echo "error: semantic capture is empty" >&2
    exit 1
fi

echo "[8/8] Run immediate CUPTI integrity queries"
{
    printf 'kernel_records='
    sqlite3 "$database" 'SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL;'
    printf 'runtime_records='
    sqlite3 "$database" 'SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME;'
    printf 'streams='
    sqlite3 "$database" 'SELECT COUNT(DISTINCT streamId) FROM CUPTI_ACTIVITY_KIND_KERNEL;'
    printf 'summed_kernel_ms='
    sqlite3 "$database" 'SELECT printf("%.6f", SUM(end-start)/1000000.0) FROM CUPTI_ACTIVITY_KIND_KERNEL;'
    printf 'kernels_without_runtime_correlation='
    sqlite3 "$database" 'SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r USING(correlationId) WHERE r.correlationId IS NULL;'
} | tee "$attempt_dir/cupti-integrity.txt"

docker stop --time 30 "$cupti_container" >/dev/null 2>&1 || true
docker rm -f "$cupti_container" >/dev/null 2>&1 || true
trap - EXIT INT TERM

printf '\nCapture complete; the GPU server is stopped.\n'
printf 'Raw evidence: %s\n' "$attempt_dir"
tail -n 1 "$attempt_dir/semantic-capture.log"
