#!/usr/bin/env bash
set -euo pipefail

root=/home/harsh4786/gpu-observer
image=nvcr.io/nvidia/vllm:26.05-py3
model=Qwen/Qwen3-14B
revision=40c069824f4251a91eefaf281ebe4c544efd3e18
inputs="$root/benchmarks/mixed-prefill/exp0012-20260811T143000Z/controlled-workload/interactive-payloads-corrected.jsonl"
aiperf=/home/harsh4786/.venvs/gpu-observer-aiperf-0.10.0/bin/aiperf
semantic_lib="$root/target/release/libgpu_observer_vllm_bridge.so"
semantic_capture="$root/target/release/semantic_capture"
semantic_stats="$root/target/release/semantic_step_stats"
semantic_dump="$root/target/release/semantic_dump"
packing_runner="$root/vllm-adapter/packed-overlay/vllm/v1/worker/gpu_model_runner.py"
semantic_core="$root/vllm-adapter/overlay/vllm/v1/engine/core.py"
semantic_py="$root/vllm-adapter/gpu_observer_semantic.py"
port=8000
run_id=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$root/benchmarks/industry-stress/exp0017-sharegpt-packed-$run_id"
container="go-aiperf-packed-$run_id"
mkdir -p "$run_dir/semantic-shm" "$run_dir/aiperf"
chmod 0777 "$run_dir" "$run_dir/semantic-shm" "$run_dir/aiperf"

request_jobs=()
logs_pid=
monitor_pid=
cleanup() {
  if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
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

for required in "$inputs" "$aiperf" "$semantic_lib" "$semantic_capture" \
                "$semantic_stats" "$semantic_dump" "$packing_runner" \
                "$semantic_core" "$semantic_py"; do
  [[ -f "$required" ]] || { echo "missing required artifact: $required" >&2; exit 1; }
done

{
  echo "experiment=EXP-0017"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "question=Can production async vLLM preserve authoritative scheduler-to-packed-row ownership under a 100-request AIPerf ShareGPT workload?"
  echo "model=$model"
  echo "revision=$revision"
  echo "image=$image"
  echo "execution_mode=FULL_AND_PIECEWISE_CUDA_GRAPHS"
  echo "async_scheduling=production_default_must_validate_true"
  echo "prefix_caching=false"
  echo "aiperf_version=$($aiperf --version)"
  echo "dataset=frozen_real_ShareGPT_raw_payload_jsonl"
  echo "dataset_entries=12"
  echo "dataset_model_input_tokens_max=90"
  echo "warmup_requests=8"
  echo "measured_requests=100"
  echo "concurrency=8"
  echo "output_tokens_per_request=128"
  echo "semantic_abi_version=2"
  echo "ownership=authoritative_packed_layout_required"
  uname -a
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,power.draw --format=csv,noheader
  docker image inspect "$image" --format 'image_id={{.Id}} image_digests={{json .RepoDigests}}'
  sha256sum "$inputs" "$semantic_lib" "$semantic_core" "$semantic_py" "$packing_runner"
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
  -v /home/harsh4786/.cache/huggingface:/root/.cache/huggingface \
  -v "$semantic_lib:/observer/libgpu_observer_vllm_bridge.so:ro" \
  -v "$semantic_core:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py:ro" \
  -v "$semantic_py:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/gpu_observer_semantic.py:ro" \
  -v "$packing_runner:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py:ro" \
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
monitor_pid=

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

engine_host_pid=$(docker top "$container" -eo pid,comm,args | awk '$2 ~ /^VLLM::EngineCor/ {print $1; exit}')
[[ -n "$engine_host_pid" ]] || { echo "could not resolve EngineCore host PID" >&2; exit 1; }
echo "$engine_host_pid" > "$run_dir/engine-host-pid.txt"
rg -F "Asynchronous scheduling is enabled" "$run_dir/server.log" > "$run_dir/async-scheduling-evidence.txt"
rg "FULL_AND_PIECEWISE" "$run_dir/server.log" > "$run_dir/cuda-graph-evidence.txt"

{
  echo "realtime_ns,mem_available_kb,engine_cpu_pct,engine_rss_kb,engine_threads,gpu_util_pct,gpu_temp_c,gpu_power_w"
  while docker ps --format '{{.Names}}' | grep -Fxq "$container"; do
    realtime_ns=$(date +%s%N)
    mem_available_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    read -r engine_cpu_pct engine_rss_kb engine_threads < <(ps -p "$engine_host_pid" -o %cpu=,rss=,nlwp= | awk 'NF == 3 {print $1,$2,$3}')
    gpu_csv=$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)
    echo "$realtime_ns,$mem_available_kb,${engine_cpu_pct:-NA},${engine_rss_kb:-NA},${engine_threads:-NA},${gpu_csv:-NA,NA,NA}"
    sleep 1
  done
} > "$run_dir/system.csv" &
monitor_pid=$!

aiperf_command=(
  "$aiperf" profile
  --model "$model"
  --url "http://127.0.0.1:$port"
  --endpoint-type chat
  --streaming
  --input-file "$inputs"
  --custom-dataset-type raw-payload
  --dataset-sampling-strategy sequential
  --warmup-request-count 8
  --request-count 100
  --concurrency 8
  --artifact-dir "$run_dir/aiperf"
)
printf '%q ' "${aiperf_command[@]}" > "$run_dir/aiperf-command.txt"
printf '\n' >> "$run_dir/aiperf-command.txt"
set +e
"${aiperf_command[@]}" > "$run_dir/aiperf.log" 2>&1
aiperf_status=$?
set -e
echo "$aiperf_status" > "$run_dir/aiperf-exit-code.txt"
[[ "$aiperf_status" == 0 ]] || { tail -100 "$run_dir/aiperf.log" >&2; exit 1; }

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=

"$semantic_capture" "$run_dir/semantic-shm/semantic.ring" \
  "$run_dir/semantic.bin" 1 > "$run_dir/semantic-capture.log" 2>&1
"$semantic_stats" "$run_dir/semantic.bin" > "$run_dir/semantic-step-stats.tsv"
"$semantic_dump" "$run_dir/semantic.bin" > "$run_dir/semantic-dump.log" 2> "$run_dir/semantic-dump.stderr"
report=$(find "$run_dir/aiperf" -type f -name profile_export_aiperf.json -print -quit)
[[ -n "$report" ]] || { echo "AIPerf report missing" >&2; exit 1; }
echo "$report" > "$run_dir/aiperf-report-path.txt"
jq -e '.request_count.avg == 100 and (.error_summary | length) == 0 and .output_sequence_length.count == 100 and .output_sequence_length.min == 128 and .output_sequence_length.max == 128 and .total_output_tokens.avg == 12800' "$report" > "$run_dir/aiperf-integrity.json"
begins=$(rg -c '^begin ' "$run_dir/semantic-dump.log")
packed_begins=$(rg -c '^packed_begin ' "$run_dir/semantic-dump.log")
ends=$(rg -c '^end ' "$run_dir/semantic-dump.log")
slices=$(rg -c '^slice ' "$run_dir/semantic-dump.log")
packed_slices=$(rg -c '^packed_slice ' "$run_dir/semantic-dump.log")
[[ "$begins" -gt 0 && "$begins" -eq "$packed_begins" && "$begins" -eq "$ends" && "$slices" -eq "$packed_slices" ]] || { echo "packed semantic count mismatch" >&2; exit 1; }
awk '/^begin / {for(i=1;i<=NF;i++) if($i ~ /^scheduled=/){split($i,a,"="); scheduled+=a[2]}} /^packed_begin / {for(i=1;i<=NF;i++) if($i ~ /^tokens=/){split($i,a,"="); packed+=a[2]}} END{exit(scheduled == 0 || scheduled != packed)}' "$run_dir/semantic-dump.log"
head -1 "$run_dir/semantic-step-stats.tsv" | rg "sequence_gaps=0 loss_markers=0 max_inflight=2" > "$run_dir/semantic-quality.txt"
if rg -i "GPU observer semantic emitter disabled|EngineCore encountered a fatal|CUDA error|misaligned address" "$run_dir/server.log"; then
  echo "fatal server or semantic-emitter evidence" >&2
  exit 1
fi

docker logs "$container" > "$run_dir/server-final.log" 2>&1 || true
docker stop --time 30 "$container" >/dev/null
wait "$logs_pid" 2>/dev/null || true
logs_pid=
monitor_pid=
docker rm "$container" >/dev/null
