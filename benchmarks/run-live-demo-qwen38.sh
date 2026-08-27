#!/usr/bin/env bash
# Live-demo container for Qwen/Qwen3.8-27B -- the generic-kernel-feed UI's
# backend (ui-qwen38/, built from ui-qwen38-wasm/). Modeled directly on
# run-live-demo-cupti.sh: same semantic-ring + CUPTI-activity overlay
# mounts, same CUPTI agent .so, same smoke-test discipline. Differences:
#
#   - model/revision: Qwen3.8-27B instead of Qwen3-14B.
#   - --enforce-eager: this model's first-ever boot on this hardware (a
#     hybrid linear-attention/full-attention architecture -- see the memory
#     gpu-observer-qwen38-migration) was validated with CUDA graphs off;
#     revisit once that's proven stable over more runs.
#   - --gpu-memory-utilization 0.6: confirmed live in a plain validation
#     boot this session (~62-64GB peak on this 121GB unified-memory box,
#     nvidia-smi's own memory fields report "Not Supported" here, so
#     `free -h` was the real signal, not nvidia-smi).
#   - Ports offset by +1000 from the Qwen3-14B stack (9000/9188-9191
#     instead of 8000/8088-8090) so the two stacks never collide, even
#     though this box's memory means they can't run concurrently anyway.
#   - No shadow/sanitizer container: the SM-occupancy feature depends on a
#     crash-validated anchor kernel that does not exist for this model yet
#     (would need the from-zero EXP-0016-style validation cycle) -- out of
#     scope for the generic-feed UI, which never shows that panel at all.
set -euo pipefail

root=/home/harsh4786/gpu-observer
image=gpu-observer/vllm-sanitizer:cuda13.2
model=Qwen/Qwen3.8-27B
revision=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
port=9000
ws_port=9190
cupti_port=9191
ui_port=9188

semantic_lib="$root/target/release/libgpu_observer_vllm_bridge.so"
semantic_stream_server="$root/target/release/semantic_stream_server"
cupti_stream_server="$root/target/release/cupti_stream_server"
semantic_core="$root/vllm-adapter/overlay/vllm/v1/engine/core.py"
semantic_py="$root/vllm-adapter/gpu_observer_semantic.py"
semantic_hash="$root/vllm-adapter/gpu_observer_hash.py"
packing_runner="$root/vllm-adapter/packed-overlay/vllm/v1/worker/gpu_model_runner.py"
frontend_serving="$root/vllm-adapter/frontend-overlay/vllm/entrypoints/openai/chat_completion/serving.py"
query_capture="$root/vllm-adapter/gpu_observer_query_capture.py"
cupti_lib="$root/cupti-agent/activity/build-live/libgpu_observer_cupti_activity.so"

run_id=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$root/benchmarks/live-demo/$run_id-qwen38"
container="go-live-demo-qwen38-$run_id"
mkdir -p "$run_dir/semantic-shm"
chmod 0777 "$run_dir" "$run_dir/semantic-shm"

for required in "$semantic_lib" "$semantic_stream_server" "$cupti_stream_server" \
                "$semantic_core" "$semantic_py" "$semantic_hash" "$packing_runner" \
                "$frontend_serving" "$query_capture" "$cupti_lib"; do
  [[ -f "$required" ]] || { echo "missing required artifact: $required" >&2; exit 1; }
done

semantic_pid=
cupti_pid=
cleanup() {
  for pid in "$semantic_pid" "$cupti_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  if docker ps --format '{{.Names}}' | grep -Fxq "$container"; then
    docker stop --timeout 10 "$container" >/dev/null 2>&1 || true
  fi
  docker rm "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

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
  -e LD_PRELOAD=/observer-cupti/libgpu_observer_cupti_activity.so \
  -e GPU_OBSERVER_CUPTI_DEFER=1 \
  -e GPU_OBSERVER_CUPTI_OUTPUT_PREFIX=/results/cupti-%p \
  -v /home/harsh4786/.cache/huggingface:/root/.cache/huggingface \
  -v "$semantic_lib:/observer/libgpu_observer_vllm_bridge.so:ro" \
  -v "$semantic_core:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py:ro" \
  -v "$semantic_py:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/gpu_observer_semantic.py:ro" \
  -v "$semantic_hash:/usr/local/lib/python3.12/dist-packages/vllm/gpu_observer_hash.py:ro" \
  -v "$packing_runner:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py:ro" \
  -v "$frontend_serving:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py:ro" \
  -v "$query_capture:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/gpu_observer_query_capture.py:ro" \
  -v "$cupti_lib:/observer-cupti/libgpu_observer_cupti_activity.so:ro" \
  -v "$run_dir/semantic-shm:/observer-shm" \
  -v "$run_dir:/results" \
  "$image" \
  vllm serve "$model" \
  --revision "$revision" \
  --port "$port" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.6 \
  --enforce-eager \
  --trust-remote-code \
  --max-num-seqs 32 \
  --no-enable-prefix-caching > "$run_dir/container-id.txt"
docker logs -f "$container" > "$run_dir/server.log" 2>&1 &

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
[[ "$engine_pid" =~ ^[0-9]+$ ]] || { echo "could not resolve EngineCore PID" >&2; exit 1; }
echo "$engine_pid" > "$run_dir/engine-container-pid.txt"

docker exec -e LD_PRELOAD= "$container" test -p "/tmp/gpu-observer-cupti-$engine_pid.fifo"
docker exec -e LD_PRELOAD= "$container" sh -c "printf S > /tmp/gpu-observer-cupti-$engine_pid.fifo"
cupti_started=0
for _ in $(seq 1 100); do
  if grep -Fq "gpu-observer-cupti: control command=S status=0" "$run_dir/server.log"; then
    cupti_started=1
    break
  fi
  sleep 0.1
done
[[ "$cupti_started" == 1 ]] || { echo "CUPTI deferred start failed" >&2; tail -50 "$run_dir/server.log" >&2; exit 1; }
cupti_activities="$run_dir/cupti-$engine_pid.activities.tsv"

"$semantic_stream_server" "$run_dir/semantic-shm/semantic.ring" "127.0.0.1:$ws_port" \
  > "$run_dir/semantic-stream-server.log" 2>&1 &
semantic_pid=$!
sleep 1
kill -0 "$semantic_pid" 2>/dev/null || { echo "semantic_stream_server failed to start" >&2; cat "$run_dir/semantic-stream-server.log" >&2; exit 1; }

"$cupti_stream_server" "$cupti_activities" "127.0.0.1:$cupti_port" \
  > "$run_dir/cupti-stream-server.log" 2>&1 &
cupti_pid=$!
sleep 1
kill -0 "$cupti_pid" 2>/dev/null || { echo "cupti_stream_server failed to start" >&2; cat "$run_dir/cupti-stream-server.log" >&2; exit 1; }

cat <<EOF

vLLM ready on http://127.0.0.1:$port (Qwen3.8-27B, CUPTI-patched, unfiltered)
semantic_stream_server tailing the ring on ws://127.0.0.1:$ws_port
cupti_stream_server tailing $cupti_activities on ws://127.0.0.1:$cupti_port

In a separate terminal, serve the WASM UI from the repo root:
  python3 -m http.server $ui_port --directory $root/ui-qwen38

Then open:
  http://127.0.0.1:$ui_port/index.html

If connecting from another machine over SSH:
  ssh -fN -o ExitOnForwardFailure=yes -L $ui_port:127.0.0.1:$ui_port -L $ws_port:127.0.0.1:$ws_port -L $cupti_port:127.0.0.1:$cupti_port -L $port:127.0.0.1:$port <host>

IMPORTANT: before connecting a browser, run the standing smoke test (one
curl chat request, then grep server.log for "misaligned address" and
"EngineCore encountered a fatal" -- must find nothing) -- this is this
model's first time running with the CUPTI/semantic overlay attached, not
just the plain validation boot from earlier.

Logs: $run_dir/server.log , $run_dir/semantic-stream-server.log , $run_dir/cupti-stream-server.log
Ctrl+C to stop the server and both stream tails.
EOF

wait "$semantic_pid" "$cupti_pid"
