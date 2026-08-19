#!/usr/bin/env bash
set -uo pipefail

audit=/home/harsh4786/gpu-observer/benchmarks/device-ebpf/gpu-ext-port-audit-20260810
fork=/home/harsh4786/gpu_ext-v0.0.21/kernel-module/nvidia-module
log="$audit/audit-systemd.log"

exec >"$log" 2>&1

date -u +started_utc=%Y-%m-%dT%H:%M:%SZ

echo section=running_modules
for module_name in nvidia nvidia_uvm nvidia_modeset nvidia_drm; do
    printf '%s ' "$module_name"
    cat "/sys/module/$module_name/version" 2>/dev/null || echo not_loaded
done

echo section=module_metadata
modinfo nvidia 2>/dev/null | sed -n '1,30p'
modinfo nvidia_uvm 2>/dev/null | sed -n '1,30p'

echo section=packages
dpkg-query -W -f='${Package}\t${Version}\n' 2>/dev/null \
    | grep -E '^(nvidia|libnvidia|cuda-drivers|linux-modules-nvidia)' \
    | sort || true

echo section=local_sources
find /usr/src -maxdepth 3 -type f \
    \( -name version.mk -o -name nv-linux.h -o -name uvm_linux.h \) \
    -printf '%h\n' 2>/dev/null | sort -u

echo section=gpu_ext_identity
git -C "$fork" remote -v
git -C "$fork" status --short
git -C "$fork" log --oneline --decorate -40
git -C "$fork" tag -l | tail -100

echo section=gpu_ext_commit_stats
git -C "$fork" log --format='%H %P %s' -20
git -C "$fork" log --stat --oneline -12

echo section=gpu_ext_files
find "$fork/kernel-open" -type f \
    \( -iname '*bpf*' -o -iname '*sched*' \) -printf '%P\n' | sort

echo section=gpu_ext_hooks
rg -n 'uvm_bpf|gpu_sched|struct_ops|register_bpf|__bpf_kfunc' \
    "$fork/kernel-open" "$fork/Makefile" 2>/dev/null || true

echo section=official_refs
timeout 120 git ls-remote --tags \
    https://github.com/NVIDIA/open-gpu-kernel-modules.git \
    'refs/tags/575.57.08' 'refs/tags/580.173.02' || true

date -u +finished_utc=%Y-%m-%dT%H:%M:%SZ
echo audit_status=complete
