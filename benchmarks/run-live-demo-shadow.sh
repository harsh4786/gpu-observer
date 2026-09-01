#!/usr/bin/env bash
# Container 2 ("shadow") of the two-container live GPU-execution-hierarchy
# demo: a reparameterized copy of run-live-demo-with-kernel-trace.sh (same
# validated sanitizer config, same semantic-ring instrumentation) running
# on separate ports (:8001/:8189/:8190) alongside container 1
# (run-live-demo-cupti.sh, :8000/:8089/:8090).
#
# Why a second container instead of one: CUPTI and Compute Sanitizer cannot
# share a process on this stack (single CUPTI subscriber slot -- see the
# plan / cupti-agent/README.md's "Join semantics" section). The browser
# mirrors every chat request to both containers (ui/trace.js's
# sendChatMessage()); this container's reply is never shown, only its GPU
# telemetry (live SM/block data for reshape_and_cache_flash_kernel, plus
# live request attribution) is consumed, by the kernel-activity card.
#
# --gpu-memory-utilization 0.4 is required, not cosmetic: without it vLLM's
# default (~0.9) computes its reservation against TOTAL device memory
# regardless of what container 1 already holds, and fails outright even
# though the actual free memory would otherwise be enough (confirmed this
# session -- see the plan's unified-memory finding).
set -euo pipefail

root=/home/harsh4786/gpu-observer
image=gpu-observer/vllm-sanitizer:cuda13.2
model=Qwen/Qwen3-14B
revision=40c069824f4251a91eefaf281ebe4c544efd3e18
port=8001
ws_port=8189
san_port=8190
ui_port=8088

semantic_lib="$root/target/release/libgpu_observer_vllm_bridge.so"
semantic_stream_server="$root/target/release/semantic_stream_server"
sanitizer_stream_server="$root/target/release/sanitizer_stream_server"
semantic_core="$root/vllm-adapter/overlay/vllm/v1/engine/core.py"
semantic_py="$root/vllm-adapter/gpu_observer_semantic.py"
semantic_hash="$root/vllm-adapter/gpu_observer_hash.py"
attention_observer="$root/vllm-adapter/gpu_observer_attention.py"
packing_runner="$root/vllm-adapter/packed-overlay/vllm/v1/worker/gpu_model_runner.py"
frontend_serving="$root/vllm-adapter/frontend-overlay/vllm/entrypoints/openai/chat_completion/serving.py"
query_capture="$root/vllm-adapter/gpu_observer_query_capture.py"
probe_build="$root/device-probes/compute-sanitizer/build-live-kernel-trace"

run_id=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$root/benchmarks/live-demo/$run_id-shadow"
container="go-live-demo-shadow-$run_id"
mkdir -p "$run_dir/semantic-shm"
chmod 0777 "$run_dir" "$run_dir/semantic-shm"

for required in "$semantic_lib" "$semantic_stream_server" "$sanitizer_stream_server" \
                "$semantic_core" "$semantic_py" "$semantic_hash" "$attention_observer" "$packing_runner" \
                "$frontend_serving" "$query_capture" \
                "$probe_build/libgpu_observer_sanitizer.so" \
                "$probe_build/gpu_observer_sanitizer_patches.cubin"; do
  [[ -f "$required" ]] || { echo "missing required artifact: $required" >&2; exit 1; }
done

semantic_pid=
sanitizer_pid=
cleanup() {
  for pid in "$semantic_pid" "$sanitizer_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  if docker ps --format '{{.Names}}' | grep -Fxq "$container"; then
    docker stop --time 10 "$container" >/dev/null 2>&1 || true
  fi
  docker rm "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

{
  echo "sanitizer_probe_build=$probe_build"
  sha256sum "$probe_build/libgpu_observer_sanitizer.so" "$probe_build/gpu_observer_sanitizer_patches.cubin"
} > "$run_dir/kernel-trace-manifest.txt"

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
  -e LD_PRELOAD=/observer-sanitizer/libgpu_observer_sanitizer.so \
  -e GPU_OBSERVER_SAN_MODE=block_event \
  -e GPU_OBSERVER_SAN_PATCH_FILE=/observer-sanitizer/gpu_observer_sanitizer_patches.cubin \
  -e GPU_OBSERVER_SAN_OUTPUT_PREFIX=/results/device-probe \
  -e GPU_OBSERVER_SAN_EVENT_CAPACITY=1048576 \
  -e GPU_OBSERVER_SAN_KERNEL_SUBSTRING=reshape_and_cache_flash_kernel \
  -e GPU_OBSERVER_SAN_CALLBACK_DATA_SCOPE=function \
  -e GPU_OBSERVER_SAN_GRAPH_NODES=1 \
  -v /home/harsh4786/.cache/huggingface:/root/.cache/huggingface \
  -v "$semantic_lib:/observer/libgpu_observer_vllm_bridge.so:ro" \
  -v "$semantic_core:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py:ro" \
  -v "$semantic_py:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/gpu_observer_semantic.py:ro" \
  -v "$semantic_hash:/usr/local/lib/python3.12/dist-packages/vllm/gpu_observer_hash.py:ro" \
  -v "$attention_observer:/usr/local/lib/python3.12/dist-packages/vllm/gpu_observer_attention.py:ro" \
  -v "$packing_runner:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py:ro" \
  -v "$frontend_serving:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py:ro" \
  -v "$query_capture:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/gpu_observer_query_capture.py:ro" \
  -v "$probe_build:/observer-sanitizer:ro" \
  -v "$run_dir/semantic-shm:/observer-shm" \
  -v "$run_dir:/results" \
  "$image" \
  vllm serve "$model" \
  --revision "$revision" \
  --port "$port" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.4 \
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
semantic_pid=$!
sleep 1
kill -0 "$semantic_pid" 2>/dev/null || { echo "semantic_stream_server failed to start" >&2; cat "$run_dir/semantic-stream-server.log" >&2; exit 1; }

"$sanitizer_stream_server" "$container" "$run_dir/device-probe.events.bin" "127.0.0.1:$san_port" \
  > "$run_dir/sanitizer-stream-server.log" 2>&1 &
sanitizer_pid=$!
sleep 1
kill -0 "$sanitizer_pid" 2>/dev/null || { echo "sanitizer_stream_server failed to start" >&2; cat "$run_dir/sanitizer-stream-server.log" >&2; exit 1; }

cat <<EOF

Shadow vLLM ready on http://127.0.0.1:$port (sanitizer-patched: reshape_and_cache_flash_kernel)
shadow semantic_stream_server tailing the ring on ws://127.0.0.1:$ws_port
shadow sanitizer_stream_server tailing device-probe.events.bin on ws://127.0.0.1:$san_port

This container's chat replies are never shown in the UI -- ui/trace.js
mirrors every request here purely to generate GPU telemetry alongside
container 1 (run-live-demo-cupti.sh). Run both scripts together.

If connecting from another machine over SSH, add these to the tunnel:
  -L $ws_port:127.0.0.1:$ws_port -L $san_port:127.0.0.1:$san_port -L $port:127.0.0.1:$port

IMPORTANT: run the smoke test from the plan (one curl chat request, then
grep server.log for "misaligned address" / "EngineCore encountered a fatal"
-- must find nothing) before connecting a browser.

Logs: $run_dir/server.log , $run_dir/semantic-stream-server.log , $run_dir/sanitizer-stream-server.log
Ctrl+C to stop the shadow server and both stream tails.
EOF

wait "$semantic_pid" "$sanitizer_pid"
