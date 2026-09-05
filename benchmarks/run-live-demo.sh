#!/usr/bin/env bash
# Starts a persistent, interactive Qwen3-14B vLLM server wired for the live
# UI: engine-step/scheduler/packed-row semantics unconditional, per-token
# focus dynamic (set by the frontend at request admission -- see
# vllm-adapter/gpu_observer_query_capture.py's set_live_focus), and a
# semantic_stream_server WebSocket tail for ui/trace.html to consume.
#
# Unlike the other benchmarks/run-*.sh scripts, this is not a bounded,
# measured experiment: it runs until you Ctrl+C, matching interactive chat
# use rather than a fixed request count. It does not run AIPerf and does not
# validate anything -- see run-aiperf-sharegpt-packed-production.sh and
# run-join-b-production-gate.sh for the measured/validated experiment shape.
set -euo pipefail

root=/home/harsh4786/gpu-observer
image=nvcr.io/nvidia/vllm:26.05-py3
model=Qwen/Qwen3-14B
revision=40c069824f4251a91eefaf281ebe4c544efd3e18
port=8000
ws_port=8089
ui_port=8088

semantic_lib="$root/target/release/libgpu_observer_vllm_bridge.so"
semantic_stream_server="$root/target/release/semantic_stream_server"
semantic_core="$root/vllm-adapter/overlay/vllm/v1/engine/core.py"
semantic_py="$root/vllm-adapter/gpu_observer_semantic.py"
attention_observer="$root/vllm-adapter/gpu_observer_attention.py"
semantic_hash="$root/vllm-adapter/gpu_observer_hash.py"
packing_runner="$root/vllm-adapter/packed-overlay/vllm/v1/worker/gpu_model_runner.py"
frontend_serving="$root/vllm-adapter/frontend-overlay/vllm/entrypoints/openai/chat_completion/serving.py"
query_capture="$root/vllm-adapter/gpu_observer_query_capture.py"

run_id=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$root/benchmarks/live-demo/$run_id"
container="go-live-demo-$run_id"
mkdir -p "$run_dir/semantic-shm"
chmod 0777 "$run_dir" "$run_dir/semantic-shm"

for required in "$semantic_lib" "$semantic_stream_server" "$semantic_core" \
                "$semantic_py" "$semantic_hash" "$packing_runner" \
                "$frontend_serving" "$query_capture"; do
  [[ -f "$required" ]] || { echo "missing required artifact: $required" >&2; echo "run: cargo build --release --workspace" >&2; exit 1; }
done

server_pid=
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  if docker ps --format '{{.Names}}' | grep -Fxq "$container"; then
    docker stop --time 10 "$container" >/dev/null 2>&1 || true
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
  -v /home/harsh4786/.cache/huggingface:/root/.cache/huggingface \
  -v "$semantic_lib:/observer/libgpu_observer_vllm_bridge.so:ro" \
  -v "$semantic_core:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py:ro" \
  -v "$semantic_py:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/gpu_observer_semantic.py:ro" \
  -v "$attention_observer:/usr/local/lib/python3.12/dist-packages/vllm/gpu_observer_attention.py:ro" \
  -v "$semantic_hash:/usr/local/lib/python3.12/dist-packages/vllm/gpu_observer_hash.py:ro" \
  -v "$packing_runner:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py:ro" \
  -v "$frontend_serving:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py:ro" \
  -v "$query_capture:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/gpu_observer_query_capture.py:ro" \
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

"$semantic_stream_server" "$run_dir/semantic-shm/semantic.ring" "127.0.0.1:$ws_port" \
  > "$run_dir/semantic-stream-server.log" 2>&1 &
server_pid=$!
sleep 1
kill -0 "$server_pid" 2>/dev/null || { echo "semantic_stream_server failed to start" >&2; cat "$run_dir/semantic-stream-server.log" >&2; exit 1; }

cat <<EOF

vLLM ready on http://127.0.0.1:$port
semantic_stream_server tailing the ring on ws://127.0.0.1:$ws_port

In a separate terminal, serve the UI from the repo root:
  python3 -m http.server $ui_port --directory $root

Then open (defaults already match the ports above):
  http://127.0.0.1:$ui_port/ui/trace.html?v=graph2

If connecting from another machine over SSH:
  ssh -fN -o ExitOnForwardFailure=yes -L $ui_port:127.0.0.1:$ui_port -L $ws_port:127.0.0.1:$ws_port -L $port:127.0.0.1:$port <host>

Logs: $run_dir/server.log , $run_dir/semantic-stream-server.log
Ctrl+C to stop both the server and the stream tail.
EOF

wait "$server_pid"
