#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/env.sh"
source "$script_dir/active-run.env"
cd "$GO_ROOT"

capture_seconds=20
attempt_id=$(date -u +%Y%m%dT%H%M%SZ)
attempt_dir="$GO_RUN/join-a-$attempt_id"
mkdir -p "$attempt_dir"

semantic_job=""
aya_job=""
aya_container="gpu-observer-aya-$attempt_id"

cleanup() {
    if [[ -n "$semantic_job" ]] && kill -0 "$semantic_job" 2>/dev/null; then
        kill "$semantic_job" 2>/dev/null || true
    fi
    if [[ -n "$aya_job" ]] && kill -0 "$aya_job" 2>/dev/null; then
        kill "$aya_job" 2>/dev/null || true
    fi
    docker rm -f "$aya_container" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

if ! docker ps --format '{{.Names}}' | rg -qx "$GO_CONTAINER"; then
    echo "error: $GO_CONTAINER is not running" >&2
    exit 1
fi

curl -fsS -o /dev/null "http://127.0.0.1:$GO_PORT/health"
docker exec "$GO_CONTAINER" chmod 0666 /observer-shm/semantic.ring

echo "[1/6] Warm the model and drain all pre-measurement semantic records"
curl -fsS \
    -H 'Content-Type: application/json' \
    --data-binary "@$script_dir/request.json" \
    "http://127.0.0.1:$GO_PORT/v1/chat/completions" \
    -o "$attempt_dir/warmup-response.json"

"$GO_SEMANTIC_CAPTURE" \
    "$GO_RUN/semantic-shm/semantic.ring" \
    "$attempt_dir/pre-measurement-semantic.bin" \
    1 >"$attempt_dir/pre-measurement-semantic.log" 2>&1

echo "[2/6] Resolve the live EngineCore PID and its mapped libcuda"
engine_pid=$(
    docker top "$GO_CONTAINER" -eo pid,comm,args |
        awk '$2 ~ /^VLLM::EngineCor/ {print $1; exit}'
)
if [[ -z "$engine_pid" ]]; then
    echo "error: could not resolve VLLM::EngineCore host PID" >&2
    docker top "$GO_CONTAINER" -eo pid,comm,args >&2
    exit 1
fi

libcuda_in_process=$(
    docker run --rm \
        --privileged \
        --pid=host \
        --network none \
        --entrypoint sh \
        "$GO_IMAGE" \
        -lc "awk '\$NF ~ /libcuda\\.so/ {print \$NF; exit}' /proc/$engine_pid/maps"
)
if [[ -z "$libcuda_in_process" ]]; then
    echo "error: EngineCore $engine_pid has no mapped libcuda path" >&2
    exit 1
fi
libcuda_host="/proc/$engine_pid/root$libcuda_in_process"

printf '      EngineCore host PID: %s\n' "$engine_pid"
printf '      libcuda: %s\n' "$libcuda_host"

echo "[3/6] Start the semantic capture window"
"$GO_SEMANTIC_CAPTURE" \
    "$GO_RUN/semantic-shm/semantic.ring" \
    "$attempt_dir/semantic.bin" \
    "$capture_seconds" >"$attempt_dir/semantic-capture.log" 2>&1 &
semantic_job=$!
sleep 0.2
if ! kill -0 "$semantic_job" 2>/dev/null; then
    echo "error: semantic capture exited before measurement" >&2
    sed -n '1,120p' "$attempt_dir/semantic-capture.log" >&2
    exit 1
fi

echo "[4/6] Attach Aya to cuLaunchKernel and cuLaunchKernelEx"
attempt_relative=${attempt_dir#"$GO_ROOT"/}
docker run --rm \
    --name "$aya_container" \
    --privileged \
    --pid=host \
    --network none \
    -v "$GO_ROOT:/observer" \
    --entrypoint /observer/target/release/gpu-observer-host-probe \
    "$GO_IMAGE" \
    "$engine_pid" \
    "$libcuda_host" \
    /observer/host-probes/ebpf/target/bpfel-unknown-none/release/gpu-observer-host-probe \
    "$capture_seconds" \
    "/observer/$attempt_relative/cuda-launches.bin" \
    >"$attempt_dir/aya-capture.log" 2>&1 &
aya_job=$!

aya_attached=0
for _ in $(seq 1 100); do
    if rg -q '^attached pid=' "$attempt_dir/aya-capture.log" 2>/dev/null; then
        aya_attached=1
        break
    fi
    if ! kill -0 "$aya_job" 2>/dev/null; then
        break
    fi
    sleep 0.1
done
if [[ "$aya_attached" -ne 1 ]]; then
    echo "error: Aya did not attach within 10 seconds" >&2
    sed -n '1,160p' "$attempt_dir/aya-capture.log" >&2
    exit 1
fi

echo "[5/6] Send exactly one measured request while both collectors are active"
curl -fsS \
    -H 'Content-Type: application/json' \
    --data-binary "@$script_dir/request.json" \
    "http://127.0.0.1:$GO_PORT/v1/chat/completions" \
    -o "$attempt_dir/response.json"

wait "$aya_job"
aya_job=""
wait "$semantic_job"
semantic_job=""

if [[ ! -s "$attempt_dir/semantic.bin" ]]; then
    echo "error: semantic capture is empty" >&2
    exit 1
fi
if [[ ! -s "$attempt_dir/cuda-launches.bin" ]]; then
    echo "error: CUDA launch capture is empty" >&2
    exit 1
fi

echo "[6/6] Join request slices and CUDA submissions by engine-step time interval"
"$GO_JOIN_A" \
    "$attempt_dir/semantic.bin" \
    "$attempt_dir/cuda-launches.bin" \
    "$engine_pid" | tee "$attempt_dir/join-a.log"

printf '\nComplete. Raw evidence: %s\n' "$attempt_dir"
printf 'Semantic capture: '
tail -n 1 "$attempt_dir/semantic-capture.log"
printf 'Aya capture: '
tail -n 1 "$attempt_dir/aya-capture.log"
