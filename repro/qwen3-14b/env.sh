#!/usr/bin/env bash
# Fixed inputs for the Qwen3-14B request -> vLLM step -> CUDA launch experiment.
# Source this file in every terminal used by the reproduction.

export GO_ROOT=/home/harsh4786/gpu-observer
export GO_IMAGE=nvcr.io/nvidia/vllm:26.05-py3
export GO_MODEL=Qwen/Qwen3-14B
export GO_MODEL_REVISION=40c069824f4251a91eefaf281ebe4c544efd3e18
export GO_HF_CACHE=/home/harsh4786/.cache/huggingface

export GO_CONTAINER=gpu-observer-repro-qwen14
export GO_PORT=8000

export GO_SEMANTIC_CORE="$GO_ROOT/vllm-adapter/overlay/vllm/v1/engine/core.py"
export GO_SEMANTIC_PY="$GO_ROOT/vllm-adapter/gpu_observer_semantic.py"
export GO_SEMANTIC_LIB="$GO_ROOT/target/release/libgpu_observer_vllm_bridge.so"
export GO_SEMANTIC_CAPTURE="$GO_ROOT/target/release/semantic_capture"
export GO_HOST_LOADER="$GO_ROOT/target/release/gpu-observer-host-probe"
export GO_HOST_EBPF="$GO_ROOT/host-probes/ebpf/target/bpfel-unknown-none/release/gpu-observer-host-probe"
export GO_JOIN_A="$GO_ROOT/target/release/join_a"

export GO_NSYS_ROOT=/opt/nvidia/nsight-systems/2025.3.2
export GO_NSYS_BIN="$GO_NSYS_ROOT/target-linux-sbsa-armv8/nsys"

export GO_BPFTIME_ROOT=/home/harsh4786/bpftime
export GO_BPFTIME_BUILD="$GO_BPFTIME_ROOT/build-gpu"
export GO_CUOBJDUMP=/usr/local/cuda-13.0/bin/cuobjdump

