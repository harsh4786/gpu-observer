#include "probe_types.h"

#include <sanitizer.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits.h>
#include <unistd.h>

namespace {

struct ContextEntry {
    std::atomic<uintptr_t> key{0};
    std::atomic<uint32_t> ready{0};
    std::atomic<uint32_t> patches_loaded{0};
    CUcontext context{};
    GoSanGlobalState* device_global{};
    GoSanKernelState* device_kernels{};
    GoSanEvent* device_events{};
};

struct FunctionEntry {
    std::atomic<uintptr_t> key{0};
    std::atomic<uint32_t> ready{0};
    CUfunction function{};
    CUmodule module{};
    uint32_t context_slot{};
    uint32_t kernel_slot{};
    std::atomic<uint64_t> launches{0};
    std::atomic<uint64_t> expected_blocks{0};
    char name[256]{};
};

struct ModuleEntry {
    std::atomic<uintptr_t> key{0};
    std::atomic<uint32_t> state{0}; // 0 unseen, 1 claimed, 2 patched, 3 failed.
};

struct ToolState {
    GoSanMode mode{GO_SAN_SUBSCRIBER};
    uint64_t capacity{65536};
    uint32_t sample_mask{1023};
    bool flush_on_sync{false};
    bool patch_all_modules{true};
    char patch_path[PATH_MAX]{};
    char output_prefix[PATH_MAX / 2]{};
    char target_substring[256]{};
    Sanitizer_SubscriberHandle subscriber{};
    std::atomic<uint64_t> modules_seen{0};
    std::atomic<uint64_t> modules_patched{0};
    std::atomic<uint64_t> module_patch_failures{0};
    std::atomic<uint64_t> launches_seen{0};
    std::atomic<uint64_t> function_table_drops{0};
    std::atomic<uint64_t> context_table_drops{0};
    std::atomic<uint64_t> callback_data_failures{0};
    std::atomic<uint64_t> setup_missed_launches{0};
    std::atomic_flag patch_api_busy = ATOMIC_FLAG_INIT;
    ContextEntry contexts[GO_SAN_MAX_CONTEXTS];
    FunctionEntry functions[GO_SAN_MAX_FUNCTIONS];
    ModuleEntry modules[GO_SAN_MAX_MODULES];
};

ToolState g_tool;
volatile sig_atomic_t g_flush_requested = 0;

const char* mode_name(GoSanMode mode)
{
    switch (mode) {
        case GO_SAN_SUBSCRIBER: return "subscriber";
        case GO_SAN_BLOCK_NOOP: return "block_noop";
        case GO_SAN_BLOCK_COUNTER: return "block_counter";
        case GO_SAN_BLOCK_EVENT: return "block_event";
        case GO_SAN_SAMPLED_MEMORY_BARRIER: return "sampled_memory_barrier";
        case GO_SAN_FULL_MEMORY: return "full_memory";
    }
    return "unknown";
}

GoSanMode parse_mode(const char* value)
{
    if (value == nullptr || std::strcmp(value, "subscriber") == 0)
        return GO_SAN_SUBSCRIBER;
    if (std::strcmp(value, "block_noop") == 0)
        return GO_SAN_BLOCK_NOOP;
    if (std::strcmp(value, "block_counter") == 0)
        return GO_SAN_BLOCK_COUNTER;
    if (std::strcmp(value, "block_event") == 0)
        return GO_SAN_BLOCK_EVENT;
    if (std::strcmp(value, "sampled_memory_barrier") == 0)
        return GO_SAN_SAMPLED_MEMORY_BARRIER;
    if (std::strcmp(value, "full_memory") == 0)
        return GO_SAN_FULL_MEMORY;
    return GO_SAN_SUBSCRIBER;
}

uint64_t parse_u64(const char* value, uint64_t fallback, uint64_t maximum)
{
    if (value == nullptr || *value == '\0')
        return fallback;
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed > maximum)
        return fallback;
    return static_cast<uint64_t>(parsed);
}

bool ok(SanitizerResult result, const char* operation)
{
    if (result == SANITIZER_SUCCESS)
        return true;
    const char* text = nullptr;
    sanitizerGetResultString(result, &text);
    std::fprintf(stderr, "gpu-observer-sanitizer: %s failed: %d (%s)\n",
                 operation, static_cast<int>(result), text ? text : "unknown");
    return false;
}

uint64_t hash_pointer(uintptr_t value)
{
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33;
    return value;
}

ContextEntry* find_context(CUcontext context)
{
    const uintptr_t key = reinterpret_cast<uintptr_t>(context);
    if (key == 0)
        return nullptr;
    for (uint32_t i = 0; i < GO_SAN_MAX_CONTEXTS; ++i) {
        if (g_tool.contexts[i].key.load(std::memory_order_acquire) == key)
            return &g_tool.contexts[i];
    }
    return nullptr;
}

uint32_t context_index(const ContextEntry* entry)
{
    return static_cast<uint32_t>(entry - g_tool.contexts);
}

ContextEntry* ensure_context(CUcontext context)
{
    if (ContextEntry* existing = find_context(context))
        return existing->ready.load(std::memory_order_acquire) == 1 ? existing : nullptr;

    const uintptr_t key = reinterpret_cast<uintptr_t>(context);
    for (uint32_t i = 0; i < GO_SAN_MAX_CONTEXTS; ++i) {
        uintptr_t empty = 0;
        ContextEntry& entry = g_tool.contexts[i];
        if (!entry.key.compare_exchange_strong(empty, key, std::memory_order_acq_rel))
            continue;

        entry.context = context;
        if (g_tool.mode != GO_SAN_SUBSCRIBER) {
            const size_t event_bytes = static_cast<size_t>(g_tool.capacity) * sizeof(GoSanEvent);
            if (!ok(sanitizerAlloc(context,
                                   reinterpret_cast<void**>(&entry.device_global),
                                   sizeof(GoSanGlobalState)), "sanitizerAlloc(global)") ||
                !ok(sanitizerAlloc(context,
                                   reinterpret_cast<void**>(&entry.device_kernels),
                                   sizeof(GoSanKernelState) * GO_SAN_MAX_FUNCTIONS),
                    "sanitizerAlloc(kernels)") ||
                !ok(sanitizerAlloc(context,
                                   reinterpret_cast<void**>(&entry.device_events),
                                   event_bytes), "sanitizerAlloc(events)")) {
                entry.ready.store(0, std::memory_order_release);
                return nullptr;
            }

            GoSanGlobalState initial{};
            initial.events = entry.device_events;
            initial.capacity = g_tool.capacity;
            initial.mode = static_cast<uint32_t>(g_tool.mode);
            initial.sample_mask = g_tool.sample_mask;
            ok(sanitizerMemset(entry.device_kernels, 0,
                               sizeof(GoSanKernelState) * GO_SAN_MAX_FUNCTIONS, nullptr),
               "sanitizerMemset(kernels)");
            ok(sanitizerMemset(entry.device_events, 0, event_bytes, nullptr),
               "sanitizerMemset(events)");
            ok(sanitizerMemcpyHostToDeviceAsync(entry.device_global, &initial,
                                                sizeof(initial), nullptr),
               "sanitizerMemcpyHostToDeviceAsync(global)");
            ok(sanitizerStreamSynchronize(nullptr), "sanitizerStreamSynchronize(setup)");
        }
        entry.ready.store(1, std::memory_order_release);
        return &entry;
    }
    g_tool.context_table_drops.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
}

FunctionEntry* ensure_function(const Sanitizer_LaunchData& launch,
                               uint32_t context_slot)
{
    const uintptr_t key = reinterpret_cast<uintptr_t>(launch.function);
    uint64_t index = hash_pointer(key) % GO_SAN_MAX_FUNCTIONS;
    for (uint32_t probe = 0; probe < GO_SAN_MAX_FUNCTIONS; ++probe) {
        FunctionEntry& entry = g_tool.functions[index];
        const uintptr_t found = entry.key.load(std::memory_order_acquire);
        if (found == key) {
            if (entry.ready.load(std::memory_order_acquire) == 1)
                return &entry;
            return nullptr;
        }
        if (found == 0) {
            uintptr_t empty = 0;
            if (entry.key.compare_exchange_strong(empty, key, std::memory_order_acq_rel)) {
                entry.function = launch.function;
                entry.module = launch.module;
                entry.context_slot = context_slot;
                entry.kernel_slot = static_cast<uint32_t>(index);
                if (launch.functionName != nullptr) {
                    std::strncpy(entry.name, launch.functionName, sizeof(entry.name) - 1);
                    entry.name[sizeof(entry.name) - 1] = '\0';
                } else {
                    std::strcpy(entry.name, "<unnamed>");
                }
                entry.ready.store(1, std::memory_order_release);
                return &entry;
            }
        }
        index = (index + 1) % GO_SAN_MAX_FUNCTIONS;
    }
    g_tool.function_table_drops.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
}

ModuleEntry* ensure_module_entry(CUmodule module)
{
    const uintptr_t key = reinterpret_cast<uintptr_t>(module);
    uint64_t index = hash_pointer(key) % GO_SAN_MAX_MODULES;
    for (uint32_t probe = 0; probe < GO_SAN_MAX_MODULES; ++probe) {
        ModuleEntry& entry = g_tool.modules[index];
        const uintptr_t found = entry.key.load(std::memory_order_acquire);
        if (found == key)
            return &entry;
        if (found == 0) {
            uintptr_t empty = 0;
            if (entry.key.compare_exchange_strong(empty, key, std::memory_order_acq_rel))
                return &entry;
        }
        index = (index + 1) % GO_SAN_MAX_MODULES;
    }
    return nullptr;
}

bool patch_module(CUcontext context, CUmodule module)
{
    if (g_tool.mode == GO_SAN_SUBSCRIBER)
        return true;
    ModuleEntry* entry = ensure_module_entry(module);
    if (entry == nullptr)
        return false;
    const uint32_t current = entry->state.load(std::memory_order_acquire);
    if (current == 2)
        return true;
    if (current == 1 || current == 3)
        return false;
    uint32_t unseen = 0;
    if (!entry->state.compare_exchange_strong(unseen, 1, std::memory_order_acq_rel))
        return false;
    if (g_tool.patch_api_busy.test_and_set(std::memory_order_acquire)) {
        entry->state.store(3, std::memory_order_release);
        g_tool.module_patch_failures.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    ContextEntry* context_entry = ensure_context(context);
    bool success = context_entry != nullptr;
    if (success) {
        uint32_t patch_state = context_entry->patches_loaded.load(std::memory_order_acquire);
        if (patch_state == 0) {
            uint32_t empty = 0;
            if (context_entry->patches_loaded.compare_exchange_strong(
                    empty, 1, std::memory_order_acq_rel)) {
                const bool loaded = ok(
                    sanitizerAddPatchesFromFile(g_tool.patch_path, context),
                    "sanitizerAddPatchesFromFile");
                context_entry->patches_loaded.store(loaded ? 2U : 3U,
                                                    std::memory_order_release);
                patch_state = loaded ? 2U : 3U;
            } else {
                patch_state = empty;
            }
        }
        success = patch_state == 2;
    }
    if (success) {
        switch (g_tool.mode) {
            case GO_SAN_BLOCK_NOOP:
                success &= ok(sanitizerPatchInstructions(SANITIZER_INSTRUCTION_BLOCK_ENTER,
                                                         module, "go_block_noop"),
                              "patch block noop");
                break;
            case GO_SAN_BLOCK_COUNTER:
                success &= ok(sanitizerPatchInstructions(SANITIZER_INSTRUCTION_BLOCK_ENTER,
                                                         module, "go_block_counter"),
                              "patch block counter");
                break;
            case GO_SAN_BLOCK_EVENT:
                success &= ok(sanitizerPatchInstructions(SANITIZER_INSTRUCTION_BLOCK_ENTER,
                                                         module, "go_block_event"),
                              "patch block event");
                break;
            case GO_SAN_SAMPLED_MEMORY_BARRIER:
                success &= ok(sanitizerPatchInstructions(SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS,
                                                         module, "go_global_memory_sampled"),
                              "patch sampled global memory");
                success &= ok(sanitizerPatchInstructions(SANITIZER_INSTRUCTION_SHARED_MEMORY_ACCESS,
                                                         module, "go_shared_memory_sampled"),
                              "patch sampled shared memory");
                success &= ok(sanitizerPatchInstructions(SANITIZER_INSTRUCTION_LOCAL_MEMORY_ACCESS,
                                                         module, "go_local_memory_sampled"),
                              "patch sampled local memory");
                success &= ok(sanitizerPatchInstructions(SANITIZER_INSTRUCTION_BARRIER,
                                                         module, "go_barrier_sampled"),
                              "patch sampled barrier");
                break;
            case GO_SAN_FULL_MEMORY:
                success &= ok(sanitizerPatchInstructions(SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS,
                                                         module, "go_global_memory_full"),
                              "patch full global memory");
                success &= ok(sanitizerPatchInstructions(SANITIZER_INSTRUCTION_SHARED_MEMORY_ACCESS,
                                                         module, "go_shared_memory_full"),
                              "patch full shared memory");
                success &= ok(sanitizerPatchInstructions(SANITIZER_INSTRUCTION_LOCAL_MEMORY_ACCESS,
                                                         module, "go_local_memory_full"),
                              "patch full local memory");
                break;
            case GO_SAN_SUBSCRIBER:
                break;
        }
    }
    if (success)
        success = ok(sanitizerPatchModule(module), "sanitizerPatchModule");

    g_tool.patch_api_busy.clear(std::memory_order_release);
    entry->state.store(success ? 2U : 3U, std::memory_order_release);
    if (success)
        g_tool.modules_patched.fetch_add(1, std::memory_order_relaxed);
    else
        g_tool.module_patch_failures.fetch_add(1, std::memory_order_relaxed);
    return success;
}

bool initialize_kernel_state(ContextEntry& context, FunctionEntry& function,
                             Sanitizer_StreamHandle stream)
{
    GoSanKernelState initial{};
    initial.global = context.device_global;
    initial.kernel_slot = function.kernel_slot;
    GoSanKernelState* destination = context.device_kernels + function.kernel_slot;
    return ok(sanitizerMemcpyHostToDeviceAsync(destination, &initial,
                                               sizeof(initial), stream),
              "sanitizerMemcpyHostToDeviceAsync(kernel)");
}

void flush_results()
{
    char summary_path[PATH_MAX];
    char events_path[PATH_MAX];
    std::snprintf(summary_path, sizeof(summary_path), "%s.summary.tsv", g_tool.output_prefix);
    std::snprintf(events_path, sizeof(events_path), "%s.events.bin", g_tool.output_prefix);

    FILE* summary = std::fopen(summary_path, "w");
    FILE* events_file = std::fopen(events_path, "wb");
    if (summary == nullptr || events_file == nullptr) {
        std::fprintf(stderr, "gpu-observer-sanitizer: flush open failed prefix=%s errno=%d\n",
                     g_tool.output_prefix, errno);
        if (summary) std::fclose(summary);
        if (events_file) std::fclose(events_file);
        return;
    }

    std::fprintf(summary,
                 "# mode=%s capacity=%llu sample_mask=%u buffer_bytes=%llu "
                 "modules_seen=%llu modules_patched=%llu module_patch_failures=%llu "
                 "launches_seen=%llu function_table_drops=%llu context_table_drops=%llu "
                 "callback_data_failures=%llu setup_missed_launches=%llu\n",
                 mode_name(g_tool.mode),
                 static_cast<unsigned long long>(g_tool.capacity),
                 g_tool.sample_mask,
                 static_cast<unsigned long long>(g_tool.capacity * sizeof(GoSanEvent)),
                 static_cast<unsigned long long>(g_tool.modules_seen.load()),
                 static_cast<unsigned long long>(g_tool.modules_patched.load()),
                 static_cast<unsigned long long>(g_tool.module_patch_failures.load()),
                 static_cast<unsigned long long>(g_tool.launches_seen.load()),
                 static_cast<unsigned long long>(g_tool.function_table_drops.load()),
                 static_cast<unsigned long long>(g_tool.context_table_drops.load()),
                 static_cast<unsigned long long>(g_tool.callback_data_failures.load()),
                 static_cast<unsigned long long>(g_tool.setup_missed_launches.load()));
    std::fprintf(summary,
                 "context_slot\tkernel_slot\tlaunches\texpected_block_callbacks\t"
                 "actual_callbacks\temitted\tdropped\tmodule\tfunction\n");

    for (uint32_t context_slot = 0; context_slot < GO_SAN_MAX_CONTEXTS; ++context_slot) {
        ContextEntry& context = g_tool.contexts[context_slot];
        if (context.ready.load(std::memory_order_acquire) != 1)
            continue;

        if (g_tool.mode == GO_SAN_SUBSCRIBER) {
            GoSanFileHeader header{};
            std::memcpy(header.magic, "GOSAN01", 8);
            header.version = GO_SAN_FORMAT_VERSION;
            header.context_slot = context_slot;
            header.mode = static_cast<uint32_t>(g_tool.mode);
            header.event_size = sizeof(GoSanEvent);
            std::fwrite(&header, sizeof(header), 1, events_file);
            for (uint32_t i = 0; i < GO_SAN_MAX_FUNCTIONS; ++i) {
                FunctionEntry& function = g_tool.functions[i];
                if (function.ready.load(std::memory_order_acquire) != 1 ||
                    function.context_slot != context_slot)
                    continue;
                std::fprintf(summary, "%u\t%u\t%llu\t%llu\t0\t0\t0\t0x%llx\t%s\n",
                             context_slot, function.kernel_slot,
                             static_cast<unsigned long long>(function.launches.load()),
                             static_cast<unsigned long long>(function.expected_blocks.load()),
                             static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(function.module)),
                             function.name);
            }
            continue;
        }

        GoSanGlobalState global{};
        auto* kernel_snapshot = static_cast<GoSanKernelState*>(
            std::calloc(GO_SAN_MAX_FUNCTIONS, sizeof(GoSanKernelState)));
        if (kernel_snapshot == nullptr)
            continue;
        if (!ok(sanitizerMemcpyDeviceToHost(&global, context.device_global,
                                            sizeof(global), nullptr),
                "flush global") ||
            !ok(sanitizerMemcpyDeviceToHost(kernel_snapshot, context.device_kernels,
                                            sizeof(GoSanKernelState) * GO_SAN_MAX_FUNCTIONS,
                                            nullptr), "flush kernels")) {
            std::free(kernel_snapshot);
            continue;
        }

        const uint64_t retained = std::min<uint64_t>(global.write_attempts, global.capacity);
        GoSanFileHeader header{};
        std::memcpy(header.magic, "GOSAN01", 8);
        header.version = GO_SAN_FORMAT_VERSION;
        header.context_slot = context_slot;
        header.mode = static_cast<uint32_t>(g_tool.mode);
        header.event_size = sizeof(GoSanEvent);
        header.capacity = global.capacity;
        header.retained_events = retained;
        header.write_attempts = global.write_attempts;
        header.dropped = global.dropped;
        header.buffer_bytes = global.capacity * sizeof(GoSanEvent);
        std::fwrite(&header, sizeof(header), 1, events_file);

        if (retained != 0) {
            const size_t bytes = static_cast<size_t>(retained) * sizeof(GoSanEvent);
            auto* event_snapshot = static_cast<GoSanEvent*>(std::malloc(bytes));
            if (event_snapshot != nullptr &&
                ok(sanitizerMemcpyDeviceToHost(event_snapshot, context.device_events,
                                               bytes, nullptr), "flush events")) {
                std::fwrite(event_snapshot, sizeof(GoSanEvent), retained, events_file);
            }
            std::free(event_snapshot);
        }

        for (uint32_t i = 0; i < GO_SAN_MAX_FUNCTIONS; ++i) {
            FunctionEntry& function = g_tool.functions[i];
            if (function.ready.load(std::memory_order_acquire) != 1 ||
                function.context_slot != context_slot)
                continue;
            const GoSanKernelState& device = kernel_snapshot[function.kernel_slot];
            std::fprintf(summary, "%u\t%u\t%llu\t%llu\t%llu\t%llu\t%llu\t0x%llx\t%s\n",
                         context_slot, function.kernel_slot,
                         static_cast<unsigned long long>(function.launches.load()),
                         static_cast<unsigned long long>(function.expected_blocks.load()),
                         static_cast<unsigned long long>(device.callback_count),
                         static_cast<unsigned long long>(device.emitted_count),
                         static_cast<unsigned long long>(device.dropped_count),
                         static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(function.module)),
                         function.name);
        }
        std::free(kernel_snapshot);
    }
    std::fflush(summary);
    std::fflush(events_file);
    fsync(fileno(summary));
    fsync(fileno(events_file));
    std::fclose(summary);
    std::fclose(events_file);
    std::fprintf(stderr, "gpu-observer-sanitizer: flushed prefix=%s\n", g_tool.output_prefix);
}

void maybe_flush()
{
    if (g_flush_requested) {
        g_flush_requested = 0;
        flush_results();
    }
}

void handle_signal(int)
{
    g_flush_requested = 1;
}

void on_module_loaded(const Sanitizer_ResourceModuleData& data)
{
    g_tool.modules_seen.fetch_add(1, std::memory_order_relaxed);
    ensure_context(data.context);
    if (g_tool.patch_all_modules)
        patch_module(data.context, data.module);
}

void on_launch(Sanitizer_CallbackId cbid, const Sanitizer_LaunchData& launch)
{
    if (cbid == SANITIZER_CBID_LAUNCH_END) {
        if (!g_tool.patch_all_modules && launch.functionName != nullptr &&
            std::strstr(launch.functionName, g_tool.target_substring) != nullptr)
            patch_module(launch.context, launch.module);
        return;
    }
    if (cbid != SANITIZER_CBID_LAUNCH_BEGIN)
        return;

    maybe_flush();
    g_tool.launches_seen.fetch_add(1, std::memory_order_relaxed);
    ContextEntry* context = ensure_context(launch.context);
    if (context == nullptr) {
        g_tool.setup_missed_launches.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    FunctionEntry* function = ensure_function(launch, context_index(context));
    if (function == nullptr) {
        g_tool.setup_missed_launches.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    function->launches.fetch_add(1, std::memory_order_relaxed);
    const uint64_t blocks = uint64_t(launch.gridDim_x) * launch.gridDim_y * launch.gridDim_z;
    function->expected_blocks.fetch_add(blocks, std::memory_order_relaxed);

    if (g_tool.mode == GO_SAN_SUBSCRIBER)
        return;

    if (!g_tool.patch_all_modules &&
        (launch.functionName == nullptr ||
         std::strstr(launch.functionName, g_tool.target_substring) == nullptr))
        return;


    GoSanKernelState* device_state = context->device_kernels + function->kernel_slot;
    if (function->launches.load(std::memory_order_relaxed) == 1 &&
        !initialize_kernel_state(*context, *function, launch.hStream)) {
        g_tool.callback_data_failures.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (!ok(sanitizerSetLaunchCallbackData(launch.hLaunch, launch.function,
                                           launch.hStream, device_state),
            "sanitizerSetLaunchCallbackData"))
        g_tool.callback_data_failures.fetch_add(1, std::memory_order_relaxed);
}

void SANITIZERAPI callback(void*, Sanitizer_CallbackDomain domain,
                           Sanitizer_CallbackId cbid, const void* cbdata)
{
    if (domain == SANITIZER_CB_DOMAIN_RESOURCE &&
        cbid == SANITIZER_CBID_RESOURCE_MODULE_LOADED) {
        on_module_loaded(*static_cast<const Sanitizer_ResourceModuleData*>(cbdata));
    } else if (domain == SANITIZER_CB_DOMAIN_LAUNCH) {
        on_launch(cbid, *static_cast<const Sanitizer_LaunchData*>(cbdata));
    } else if (domain == SANITIZER_CB_DOMAIN_SYNCHRONIZE && g_tool.flush_on_sync) {
        maybe_flush();
        if (!g_flush_requested)
            flush_results();
    }
}

int initialize()
{
    g_tool.mode = parse_mode(std::getenv("GPU_OBSERVER_SAN_MODE"));
    g_tool.capacity = parse_u64(std::getenv("GPU_OBSERVER_SAN_EVENT_CAPACITY"),
                                65536, GO_SAN_MAX_EVENTS);
    const uint64_t sample_log2 = parse_u64(std::getenv("GPU_OBSERVER_SAN_SAMPLE_LOG2"),
                                           10, 30);
    g_tool.sample_mask = sample_log2 == 0 ? 0U : ((1U << sample_log2) - 1U);
    g_tool.flush_on_sync = std::getenv("GPU_OBSERVER_SAN_FLUSH_ON_SYNC") != nullptr;

    const char* patch = std::getenv("GPU_OBSERVER_SAN_PATCH_FILE");
    const char* output = std::getenv("GPU_OBSERVER_SAN_OUTPUT_PREFIX");
    const char* target = std::getenv("GPU_OBSERVER_SAN_KERNEL_SUBSTRING");
    std::snprintf(g_tool.patch_path, sizeof(g_tool.patch_path), "%s",
                  patch ? patch : "gpu_observer_sanitizer_patches.cubin");
    std::snprintf(g_tool.output_prefix, sizeof(g_tool.output_prefix), "%s",
                  output ? output : "/tmp/gpu-observer-sanitizer");
    if (target != nullptr && *target != '\0') {
        std::snprintf(g_tool.target_substring, sizeof(g_tool.target_substring), "%s", target);
        g_tool.patch_all_modules = false;
    }

    std::signal(SIGUSR2, handle_signal);
    const SanitizerResult subscribed = sanitizerSubscribe(&g_tool.subscriber, callback, nullptr);
    if (!ok(subscribed, "sanitizerSubscribe"))
        return 1;
    ok(sanitizerEnableDomain(1, g_tool.subscriber, SANITIZER_CB_DOMAIN_RESOURCE),
       "enable resource domain");
    ok(sanitizerEnableDomain(1, g_tool.subscriber, SANITIZER_CB_DOMAIN_LAUNCH),
       "enable launch domain");
    if (g_tool.flush_on_sync)
        ok(sanitizerEnableDomain(1, g_tool.subscriber, SANITIZER_CB_DOMAIN_SYNCHRONIZE),
           "enable synchronize domain");
    std::fprintf(stderr,
                 "gpu-observer-sanitizer: initialized mode=%s capacity=%llu sample=1/%u target=%s\n",
                 mode_name(g_tool.mode),
                 static_cast<unsigned long long>(g_tool.capacity),
                 g_tool.sample_mask + 1,
                 g_tool.patch_all_modules ? "<all-modules>" : g_tool.target_substring);
    return 0;
}

} // namespace

int gpu_observer_sanitizer_initializer = initialize();

