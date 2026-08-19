#!/usr/bin/env bash
set -uo pipefail

bpftime_root=/home/harsh4786/bpftime
run_dir=/home/harsh4786/gpu-observer/benchmarks/device-ebpf/exp0007-20260810T095442Z
tracer="$bpftime_root/example/gpu/kernel_trace/kernel_trace"
sample="$run_dir/vec_add-sm121"
syscall_server="$bpftime_root/build-gpu/runtime/syscall-server/libbpftime-syscall-server.so"
agent="$bpftime_root/build-gpu/runtime/agent/libbpftime-agent.so"

export PATH=/usr/local/cuda-13.0/bin:/usr/bin:/bin
export BPFTIME_CUDA_ROOT=/usr/local/cuda-13.0
export BPFTIME_CUOBJDUMP=/usr/local/cuda-13.0/bin/cuobjdump
export BPFTIME_LOG_OUTPUT=console

BPFTIME_MAP_GPU_THREAD_COUNT=8192 \
BPFTIME_SHM_MEMORY_MB=256 \
LD_PRELOAD="$syscall_server" \
timeout --signal=TERM --kill-after=2s 14s "$tracer" \
    >"$run_dir/bpftime-tracer-sm121-ptx90.log" 2>&1 &
tracer_pid=$!

sleep 2

LD_PRELOAD="$agent" \
timeout --signal=TERM --kill-after=2s 8s "$sample" \
    >"$run_dir/bpftime-target-sm121-ptx90.log" 2>&1
target_status=$?

wait "$tracer_pid"
tracer_status=$?

printf 'target_status=%s\ntracer_status=%s\n' \
    "$target_status" "$tracer_status" \
    >"$run_dir/bpftime-process-status-sm121-ptx90.txt"

printf 'target_status=%s tracer_status=%s\n' "$target_status" "$tracer_status"
