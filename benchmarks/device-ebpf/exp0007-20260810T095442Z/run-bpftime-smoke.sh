#!/usr/bin/env bash
set -uo pipefail

bpftime_root=/home/harsh4786/bpftime
run_dir=/home/harsh4786/gpu-observer/benchmarks/device-ebpf/exp0007-20260810T095442Z
tracer="$bpftime_root/example/gpu/kernel_trace/kernel_trace"
sample="$bpftime_root/example/gpu/kernel_trace/vec_add"
syscall_server="$bpftime_root/build-gpu/runtime/syscall-server/libbpftime-syscall-server.so"
agent="$bpftime_root/build-gpu/runtime/agent/libbpftime-agent.so"

for required in "$tracer" "$sample" "$syscall_server" "$agent"; do
    if [[ ! -e "$required" ]]; then
        printf 'missing required artifact: %s\n' "$required" >&2
        exit 2
    fi
done

export PATH=/usr/local/cuda-13.0/bin:/usr/bin:/bin
export BPFTIME_CUDA_ROOT=/usr/local/cuda-13.0
export BPFTIME_CUOBJDUMP=/usr/local/cuda-13.0/bin/cuobjdump
export BPFTIME_LOG_OUTPUT=console

BPFTIME_MAP_GPU_THREAD_COUNT=8192 \
BPFTIME_SHM_MEMORY_MB=256 \
LD_PRELOAD="$syscall_server" \
timeout --signal=TERM --kill-after=2s 14s "$tracer" \
    >"$run_dir/bpftime-tracer.log" 2>&1 &
tracer_pid=$!

# The loader must finish publishing its BPF programs before the CUDA process
# registers its fatbinary. This is synchronization, not part of the benchmark.
sleep 2

LD_PRELOAD="$agent" \
timeout --signal=TERM --kill-after=2s 8s "$sample" \
    >"$run_dir/bpftime-target.log" 2>&1
target_status=$?

wait "$tracer_pid"
tracer_status=$?

# GNU timeout returns 124 when it ended an otherwise healthy infinite-loop
# sample. Preserve the values so timeout is not confused with a probe failure.
printf 'target_status=%s\ntracer_status=%s\n' \
    "$target_status" "$tracer_status" >"$run_dir/bpftime-process-status.txt"

printf 'target_status=%s tracer_status=%s\n' "$target_status" "$tracer_status"
