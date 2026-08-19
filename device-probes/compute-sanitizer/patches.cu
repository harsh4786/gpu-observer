#include "probe_types.h"

#include <sanitizer_patching.h>

static __device__ __forceinline__ uint32_t go_smid()
{
    uint32_t value;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
    return value;
}

static __device__ __forceinline__ uint64_t go_globaltimer()
{
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

static __device__ __forceinline__ uint32_t go_thread_linear()
{
    return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

static __device__ __forceinline__ bool go_block_leader()
{
    return threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0;
}

static __device__ __forceinline__ uint16_t go_memory_kind(uint16_t read_kind,
                                                          uint16_t write_kind,
                                                          uint32_t flags)
{
    return (flags & SANITIZER_MEMORY_DEVICE_FLAG_WRITE) ? write_kind : read_kind;
}

static __device__ __forceinline__ bool go_sample(const GoSanKernelState* state,
                                                 uint64_t pc)
{
    uint64_t mix = pc ^ uint64_t(go_thread_linear()) ^
                   (uint64_t(blockIdx.x) << 20) ^
                   (uint64_t(blockIdx.y) << 36) ^
                   (uint64_t(blockIdx.z) << 52);
    // MurmurHash3 finalizer: sampling must not depend on weak low address bits.
    mix ^= mix >> 33;
    mix *= 0xff51afd7ed558ccdULL;
    mix ^= mix >> 33;
    mix *= 0xc4ceb9fe1a85ec53ULL;
    mix ^= mix >> 33;
    return (mix & state->global->sample_mask) == 0;
}

static __device__ __forceinline__ void go_emit(GoSanCallbackState* callback,
                                               uint64_t pc,
                                               uint64_t address,
                                               uint32_t flags,
                                               uint16_t kind,
                                               uint16_t access_size)
{
    GoSanKernelState* state = callback->kernel;
    GoSanGlobalState* global = state->global;
    const unsigned long long sequence = atomicAdd(&global->write_attempts, 1ULL);
    if (sequence >= global->capacity) {
        atomicAdd(&global->dropped, 1ULL);
        atomicAdd(&state->dropped_count, 1ULL);
        return;
    }

    GoSanEvent& event = global->events[sequence];
    event.device_timestamp_raw = go_globaltimer();
    event.pc = pc;
    event.address = address;
    event.launch_id = callback->launch_id;
    event.sequence = static_cast<uint32_t>(sequence);
    event.kernel_slot = state->kernel_slot;
    event.block_x = blockIdx.x;
    event.block_y = blockIdx.y;
    event.block_z = blockIdx.z;
    event.thread_linear = static_cast<uint16_t>(go_thread_linear());
    event.flags = flags;
    event.kind = kind;
    event.access_size = access_size;
    event.sm_id = static_cast<uint16_t>(go_smid());
    atomicAdd(&state->emitted_count, 1ULL);
}

extern "C" __device__ __noinline__
SanitizerPatchResult go_block_noop(void*, uint64_t)
{
    return SANITIZER_PATCH_SUCCESS;
}

extern "C" __device__ __noinline__
SanitizerPatchResult go_block_counter(void* userdata, uint64_t)
{
    if (userdata != nullptr && go_block_leader()) {
        auto* callback = static_cast<GoSanCallbackState*>(userdata);
        auto* state = callback->kernel;
        atomicAdd(&state->callback_count, 1ULL);
    }
    return SANITIZER_PATCH_SUCCESS;
}

extern "C" __device__ __noinline__
SanitizerPatchResult go_block_event(void* userdata, uint64_t pc)
{
    if (userdata != nullptr && go_block_leader()) {
        auto* callback = static_cast<GoSanCallbackState*>(userdata);
        auto* state = callback->kernel;
        atomicAdd(&state->callback_count, 1ULL);
        go_emit(callback, pc, 0, 0, GO_SAN_EVENT_BLOCK, 0);
    }
    return SANITIZER_PATCH_SUCCESS;
}

static __device__ __forceinline__ SanitizerPatchResult
go_memory_sampled_common(void* userdata,
                         uint64_t pc,
                         void* ptr,
                         uint32_t access_size,
                         uint32_t flags,
                         uint16_t read_kind,
                         uint16_t write_kind)
{
    if (userdata == nullptr)
        return SANITIZER_PATCH_SUCCESS;

    auto* callback = static_cast<GoSanCallbackState*>(userdata);
    auto* state = callback->kernel;
    atomicAdd(&state->callback_count, 1ULL);
    if (go_sample(state, pc)) {
        go_emit(callback, pc, reinterpret_cast<uint64_t>(ptr), flags,
                go_memory_kind(read_kind, write_kind, flags),
                static_cast<uint16_t>(access_size));
    }
    return SANITIZER_PATCH_SUCCESS;
}

extern "C" __device__ __noinline__
SanitizerPatchResult go_global_memory_sampled(void* userdata,
                                              uint64_t pc,
                                              void* ptr,
                                              uint32_t access_size,
                                              uint32_t flags,
                                              const void*)
{
    return go_memory_sampled_common(userdata, pc, ptr, access_size, flags,
                                    GO_SAN_EVENT_GLOBAL_READ,
                                    GO_SAN_EVENT_GLOBAL_WRITE);
}

extern "C" __device__ __noinline__
SanitizerPatchResult go_shared_memory_sampled(void* userdata,
                                              uint64_t pc,
                                              void* ptr,
                                              uint32_t access_size,
                                              uint32_t flags,
                                              const void*)
{
    return go_memory_sampled_common(userdata, pc, ptr, access_size, flags,
                                    GO_SAN_EVENT_SHARED_READ,
                                    GO_SAN_EVENT_SHARED_WRITE);
}

extern "C" __device__ __noinline__
SanitizerPatchResult go_local_memory_sampled(void* userdata,
                                             uint64_t pc,
                                             void* ptr,
                                             uint32_t access_size,
                                             uint32_t flags,
                                             const void*)
{
    return go_memory_sampled_common(userdata, pc, ptr, access_size, flags,
                                    GO_SAN_EVENT_LOCAL_READ,
                                    GO_SAN_EVENT_LOCAL_WRITE);
}

extern "C" __device__ __noinline__
SanitizerPatchResult go_barrier_sampled(void* userdata,
                                       uint64_t pc,
                                       uint32_t bar_index,
                                       uint32_t thread_count,
                                       uint32_t flags)
{
    if (userdata == nullptr)
        return SANITIZER_PATCH_SUCCESS;

    auto* callback = static_cast<GoSanCallbackState*>(userdata);
    auto* state = callback->kernel;
    atomicAdd(&state->callback_count, 1ULL);
    if (go_block_leader() && go_sample(state, pc)) {
        const uint32_t packed = (bar_index & 0xffffU) | (thread_count << 16);
        go_emit(callback, pc, 0, packed | flags, GO_SAN_EVENT_BARRIER, 0);
    }
    return SANITIZER_PATCH_SUCCESS;
}

static __device__ __forceinline__ SanitizerPatchResult
go_memory_full_common(void* userdata,
                      uint64_t pc,
                      void* ptr,
                      uint32_t access_size,
                      uint32_t flags,
                      uint16_t read_kind,
                      uint16_t write_kind)
{
    if (userdata == nullptr)
        return SANITIZER_PATCH_SUCCESS;

    auto* callback = static_cast<GoSanCallbackState*>(userdata);
    auto* state = callback->kernel;
    atomicAdd(&state->callback_count, 1ULL);
    go_emit(callback, pc, reinterpret_cast<uint64_t>(ptr), flags,
            go_memory_kind(read_kind, write_kind, flags),
            static_cast<uint16_t>(access_size));
    return SANITIZER_PATCH_SUCCESS;
}

extern "C" __device__ __noinline__
SanitizerPatchResult go_global_memory_full(void* userdata,
                                           uint64_t pc,
                                           void* ptr,
                                           uint32_t access_size,
                                           uint32_t flags,
                                           const void*)
{
    return go_memory_full_common(userdata, pc, ptr, access_size, flags,
                                 GO_SAN_EVENT_GLOBAL_READ,
                                 GO_SAN_EVENT_GLOBAL_WRITE);
}

extern "C" __device__ __noinline__
SanitizerPatchResult go_shared_memory_full(void* userdata,
                                           uint64_t pc,
                                           void* ptr,
                                           uint32_t access_size,
                                           uint32_t flags,
                                           const void*)
{
    return go_memory_full_common(userdata, pc, ptr, access_size, flags,
                                 GO_SAN_EVENT_SHARED_READ,
                                 GO_SAN_EVENT_SHARED_WRITE);
}

extern "C" __device__ __noinline__
SanitizerPatchResult go_local_memory_full(void* userdata,
                                          uint64_t pc,
                                          void* ptr,
                                          uint32_t access_size,
                                          uint32_t flags,
                                          const void*)
{
    return go_memory_full_common(userdata, pc, ptr, access_size, flags,
                                 GO_SAN_EVENT_LOCAL_READ,
                                 GO_SAN_EVENT_LOCAL_WRITE);
}

