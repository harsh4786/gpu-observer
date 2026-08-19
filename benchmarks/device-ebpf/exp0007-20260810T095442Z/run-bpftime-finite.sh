#!/usr/bin/env bash
set -uo pipefail

bpftime_root=/home/harsh4786/bpftime
run_dir=/home/harsh4786/gpu-observer/benchmarks/device-ebpf/exp0007-20260810T095442Z
tracer="$bpftime_root/example/gpu/kernel_trace/kernel_trace"
sample="$run_dir/vec_add_once-sm121"
syscall_server="$bpftime_root/build-gpu/runtime/syscall-server/libbpftime-syscall-server.so"
agent="$bpftime_root/build-gpu/runtime/agent/libbpftime-agent.so"

export PATH=/usr/local/cuda-13.0/bin:/usr/bin:/bin
export BPFTIME_CUDA_ROOT=/usr/local/cuda-13.0
export BPFTIME_CUOBJDUMP=/usr/local/cuda-13.0/bin/cuobjdump
export BPFTIME_LOG_OUTPUT=console

# Put LD_PRELOAD inside `env`, after `timeout`, so bpftime initializes only in
# the intended tracer/target processes rather than in the timeout wrappers.
timeout --signal=TERM --kill-after=2s 14s env \
    BPFTIME_MAP_GPU_THREAD_COUNT=8192 \
    BPFTIME_SHM_MEMORY_MB=256 \
    LD_PRELOAD="$syscall_server" \
    "$tracer" >"$run_dir/bpftime-tracer-finite.log" 2>&1 &
tracer_pid=$!

sleep 2

timeout --signal=TERM --kill-after=2s 10s env \
    LD_PRELOAD="$agent" \
    "$sample" >"$run_dir/bpftime-target-finite.log" 2>&1
target_status=$?

wait "$tracer_pid"
tracer_status=$?

printf 'target_status=%s\ntracer_status=%s\n' \
    "$target_status" "$tracer_status" \
    >"$run_dir/bpftime-process-status-finite.txt"

printf 'target_status=%s tracer_status=%s\n' "$target_status" "$tracer_status"
