#include "probe_types.h"

#include <sanitizer.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
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
    GoSanCallbackState* device_function_callbacks{};
    GoSanCallbackState* device_launch_callbacks{};
    GoSanEvent* device_events{};
};

struct FunctionEntry {
    std::atomic<uintptr_t> key{0};
    std::atomic<uint32_t> ready{0};
    CUfunction function{};
    CUmodule module{};
    uint32_t context_slot{};
    uint32_t kernel_slot{};
    uint64_t function_pc{};
    uint64_t function_size{};
    std::atomic<uint64_t> launches{0};
    std::atomic<uint64_t> expected_blocks{0};
    char name[256]{};
};

struct ModuleEntry {
    std::atomic<uintptr_t> key{0};
    std::atomic<uint32_t> state{0}; // 0 unseen, 1 claimed, 2 patched, 3 failed.
};

struct LaunchEntry {
    std::atomic<uint32_t> ready{0};
    uint32_t context_slot{};
    uint32_t kernel_slot{};
    uint32_t reserved{};
    uint64_t launch_id{};
    uint64_t grid_id{};
    uint64_t host_timestamp_ns{};
    uintptr_t stream{};
    uint32_t grid_x{};
    uint32_t grid_y{};
    uint32_t grid_z{};
    uint32_t block_x{};
    uint32_t block_y{};
    uint32_t block_z{};
    uint64_t expected_blocks{};
};

struct GraphNodeEntry {
    std::atomic<uint32_t> ready{0};
    uint32_t context_slot{};
    uint32_t kernel_slot{};
    uint32_t graph_launch_id{};
    uint64_t grid_id{};
    uint64_t host_timestamp_ns{};
    uintptr_t graph_exec{};
    uintptr_t node{};
    uintptr_t stream{};
    uintptr_t api_stream{};
    uint32_t grid_x{};
    uint32_t grid_y{};
    uint32_t grid_z{};
    uint32_t block_x{};
    uint32_t block_y{};
    uint32_t block_z{};
    uint64_t expected_blocks{};
};

struct ToolState {
    GoSanMode mode{GO_SAN_SUBSCRIBER};
    uint64_t capacity{65536};
    uint32_t sample_mask{1023};
    bool flush_on_sync{false};
    bool patch_all_modules{true};
    bool launch_identity{false};
    bool passive_launch_records{false};
    bool function_callback_data{false};
    bool graph_node_identity{false};
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
    std::atomic<uint64_t> attributed_launches{0};
    std::atomic<uint64_t> launch_records{0};
    std::atomic<uint64_t> passive_launches_seen{0};
    std::atomic<uint64_t> capture_arms{0};
    std::atomic<uint64_t> capture_arm_failures{0};
    std::atomic<uint64_t> launch_table_drops{0};
    std::atomic<uint64_t> setup_missed_launches{0};
    std::atomic<uint64_t> graph_nodes_seen{0};
    std::atomic<uint64_t> graph_nodes_recorded{0};
    std::atomic<uint64_t> graph_node_table_drops{0};
    std::atomic_flag patch_api_busy = ATOMIC_FLAG_INIT;
    ContextEntry contexts[GO_SAN_MAX_CONTEXTS];
    FunctionEntry functions[GO_SAN_MAX_FUNCTIONS];
    ModuleEntry modules[GO_SAN_MAX_MODULES];
    LaunchEntry launches[GO_SAN_MAX_LAUNCHES];
    GraphNodeEntry graph_nodes[GO_SAN_MAX_LAUNCHES];
};

ToolState g_tool;
volatile sig_atomic_t g_flush_requested = 0;
volatile sig_atomic_t g_arm_requested = 0;
volatile sig_atomic_t g_arm_on_first_signal = 0;

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

bool env_enabled(const char* value)
{
    return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
}

uint64_t monotonic_ns()
{
    timespec now{};
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return 0;
    return uint64_t(now.tv_sec) * 1'000'000'000ULL + uint64_t(now.tv_nsec);
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
                !ok(sanitizerAlloc(
                        context,
                        reinterpret_cast<void**>(&entry.device_function_callbacks),
                        sizeof(GoSanCallbackState) * GO_SAN_MAX_FUNCTIONS),
                    "sanitizerAlloc(function callbacks)") ||
                (g_tool.launch_identity &&
                 !ok(sanitizerAlloc(
                         context,
                         reinterpret_cast<void**>(&entry.device_launch_callbacks),
                         sizeof(GoSanCallbackState) * GO_SAN_MAX_LAUNCHES),
                     "sanitizerAlloc(launch callbacks)")) ||
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
            ok(sanitizerMemset(entry.device_function_callbacks, 0,
                               sizeof(GoSanCallbackState) * GO_SAN_MAX_FUNCTIONS, nullptr),
               "sanitizerMemset(function callbacks)");
            if (g_tool.launch_identity)
                ok(sanitizerMemset(entry.device_launch_callbacks, 0,
                                   sizeof(GoSanCallbackState) * GO_SAN_MAX_LAUNCHES, nullptr),
                   "sanitizerMemset(launch callbacks)");
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
                if (launch.functionName != nullptr) {
                    ok(sanitizerGetFunctionPcAndSize(
                           entry.module, launch.functionName,
                           &entry.function_pc, &entry.function_size),
                       "sanitizerGetFunctionPcAndSize");
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
    GoSanKernelState* kernel_destination =
        context.device_kernels + function.kernel_slot;
    GoSanCallbackState callback{};
    callback.kernel = kernel_destination;
    GoSanCallbackState* callback_destination =
        context.device_function_callbacks + function.kernel_slot;
    return ok(sanitizerMemcpyHostToDeviceAsync(kernel_destination, &initial,
                                               sizeof(initial), stream),
              "sanitizerMemcpyHostToDeviceAsync(kernel)") &&
           ok(sanitizerMemcpyHostToDeviceAsync(callback_destination, &callback,
                                               sizeof(callback), stream),
              "sanitizerMemcpyHostToDeviceAsync(function callback)");
}

bool record_launch_entry(ContextEntry& context,
                         const FunctionEntry& function,
                         const Sanitizer_LaunchData& launch,
                         uint64_t expected_blocks,
                         uint64_t launch_id)
{
    const uint64_t record_index =
        g_tool.launch_records.fetch_add(1, std::memory_order_relaxed);
    if (record_index >= GO_SAN_MAX_LAUNCHES) {
        g_tool.launch_table_drops.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    LaunchEntry& record = g_tool.launches[record_index];
    record.context_slot = context_index(&context);
    record.kernel_slot = function.kernel_slot;
    record.launch_id = launch_id;
    record.grid_id = launch.gridId;
    record.host_timestamp_ns = monotonic_ns();
    record.stream = reinterpret_cast<uintptr_t>(launch.stream);
    record.grid_x = launch.gridDim_x;
    record.grid_y = launch.gridDim_y;
    record.grid_z = launch.gridDim_z;
    record.block_x = launch.blockDim_x;
    record.block_y = launch.blockDim_y;
    record.block_z = launch.blockDim_z;
    record.expected_blocks = expected_blocks;
    record.ready.store(1, std::memory_order_release);
    return true;
}

void record_passive_launch(ContextEntry& context,
                           const FunctionEntry& function,
                           const Sanitizer_LaunchData& launch,
                           uint64_t expected_blocks)
{
    const uint64_t launch_id =
        g_tool.passive_launches_seen.fetch_add(1, std::memory_order_relaxed) + 1;
    record_launch_entry(context, function, launch, expected_blocks, launch_id);
}

GoSanCallbackState* callback_state_for_launch(ContextEntry& context,
                                              const FunctionEntry& function,
                                              const Sanitizer_LaunchData& launch,
                                              uint64_t expected_blocks)
{
    if (!g_tool.launch_identity)
        return context.device_function_callbacks + function.kernel_slot;

    const uint64_t attributed_index =
        g_tool.attributed_launches.fetch_add(1, std::memory_order_relaxed);
    if (attributed_index >= GO_SAN_MAX_LAUNCHES) {
        g_tool.launch_table_drops.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    const uint64_t launch_id = attributed_index + 1;
    GoSanCallbackState initial{};
    initial.kernel = context.device_kernels + function.kernel_slot;
    initial.launch_id = launch_id;
    GoSanCallbackState* destination =
        context.device_launch_callbacks + attributed_index;
    if (!ok(sanitizerMemcpyHostToDeviceAsync(destination, &initial,
                                             sizeof(initial), launch.hStream),
            "sanitizerMemcpyHostToDeviceAsync(launch callback)"))
        return nullptr;
    if (!record_launch_entry(context, function, launch, expected_blocks, launch_id))
        return nullptr;
    return destination;
}

bool copy_device_to_host(ContextEntry&, void* host,
                         const void* device, size_t bytes, const char* operation)
{
    return ok(
        sanitizerMemcpyDeviceToHost(host, const_cast<void*>(device), bytes, nullptr),
        operation);
}
void flush_results()
{
    char summary_path[PATH_MAX];
    char events_path[PATH_MAX];
    char launches_path[PATH_MAX];
    char graph_nodes_path[PATH_MAX];
    std::snprintf(summary_path, sizeof(summary_path), "%s.summary.tsv", g_tool.output_prefix);
    std::snprintf(events_path, sizeof(events_path), "%s.events.bin", g_tool.output_prefix);
    std::snprintf(launches_path, sizeof(launches_path), "%s.launches.tsv", g_tool.output_prefix);
    std::snprintf(graph_nodes_path, sizeof(graph_nodes_path), "%s.graph-nodes.tsv", g_tool.output_prefix);

    FILE* summary = std::fopen(summary_path, "w");
    FILE* events_file = std::fopen(events_path, "wb");
    FILE* launches_file = std::fopen(launches_path, "w");
    FILE* graph_nodes_file = std::fopen(graph_nodes_path, "w");
    if (summary == nullptr || events_file == nullptr || launches_file == nullptr ||
        graph_nodes_file == nullptr) {
        std::fprintf(stderr, "gpu-observer-sanitizer: flush open failed prefix=%s errno=%d\n",
                     g_tool.output_prefix, errno);
        if (summary) std::fclose(summary);
        if (events_file) std::fclose(events_file);
        if (launches_file) std::fclose(launches_file);
        if (graph_nodes_file) std::fclose(graph_nodes_file);
        return;
    }

    std::fprintf(summary,
                 "# mode=%s capacity=%llu sample_mask=%u buffer_bytes=%llu "
                 "modules_seen=%llu modules_patched=%llu module_patch_failures=%llu "
                 "launches_seen=%llu function_table_drops=%llu context_table_drops=%llu "
                 "callback_data_failures=%llu setup_missed_launches=%llu "
                 "callback_data_scope=%s launch_identity=%u attributed_launches=%llu passive_launch_records=%u passive_launches_seen=%llu launch_records=%llu launch_table_drops=%llu "
                 "graph_node_identity=%u graph_nodes_seen=%llu graph_nodes_recorded=%llu "
                 "graph_node_table_drops=%llu graph_node_table_bytes=%zu "
                 "capture_arms=%llu capture_arm_failures=%llu\n",
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
                 static_cast<unsigned long long>(g_tool.setup_missed_launches.load()),
                 g_tool.function_callback_data ? "function" : "launch",
                 g_tool.launch_identity ? 1U : 0U,
                 static_cast<unsigned long long>(g_tool.attributed_launches.load()),
                 g_tool.passive_launch_records ? 1U : 0U,
                 static_cast<unsigned long long>(g_tool.passive_launches_seen.load()),
                 static_cast<unsigned long long>(g_tool.launch_records.load()),
                 static_cast<unsigned long long>(g_tool.launch_table_drops.load()),
                 g_tool.graph_node_identity ? 1U : 0U,
                 static_cast<unsigned long long>(g_tool.graph_nodes_seen.load()),
                 static_cast<unsigned long long>(g_tool.graph_nodes_recorded.load()),
                 static_cast<unsigned long long>(g_tool.graph_node_table_drops.load()),
                 sizeof(g_tool.graph_nodes),
                 static_cast<unsigned long long>(g_tool.capture_arms.load()),
                 static_cast<unsigned long long>(g_tool.capture_arm_failures.load()));
    std::fprintf(summary,
                 "context_slot\tkernel_slot\tlaunches\texpected_block_callbacks\t"
                 "actual_callbacks\temitted\tdropped\tfunction_pc\tfunction_size\tmodule\tfunction\n");

    for (uint32_t context_slot = 0; context_slot < GO_SAN_MAX_CONTEXTS; ++context_slot) {
        ContextEntry& context = g_tool.contexts[context_slot];
        if (context.ready.load(std::memory_order_acquire) != 1)
            continue;

        if (g_tool.mode == GO_SAN_SUBSCRIBER) {
            GoSanFileHeader header{};
            std::memcpy(header.magic, "GOSAN02", 8);
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
                std::fprintf(summary, "%u\t%u\t%llu\t%llu\t0\t0\t0\t0x%llx\t%llu\t0x%llx\t%s\n",
                             context_slot, function.kernel_slot,
                             static_cast<unsigned long long>(function.launches.load()),
                             static_cast<unsigned long long>(function.expected_blocks.load()),
                             static_cast<unsigned long long>(function.function_pc),
                             static_cast<unsigned long long>(function.function_size),
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
        if (!copy_device_to_host(context, &global, context.device_global,
                                 sizeof(global), "flush global") ||
            !copy_device_to_host(
                context, kernel_snapshot, context.device_kernels,
                sizeof(GoSanKernelState) * GO_SAN_MAX_FUNCTIONS, "flush kernels")) {
            std::free(kernel_snapshot);
            continue;
        }

        const uint64_t retained = std::min<uint64_t>(global.write_attempts, global.capacity);
        GoSanFileHeader header{};
        std::memcpy(header.magic, "GOSAN02", 8);
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
                copy_device_to_host(context, event_snapshot,
                                    context.device_events, bytes,
                                    "flush events")) {
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
            std::fprintf(summary, "%u\t%u\t%llu\t%llu\t%llu\t%llu\t%llu\t0x%llx\t%llu\t0x%llx\t%s\n",
                         context_slot, function.kernel_slot,
                         static_cast<unsigned long long>(function.launches.load()),
                         static_cast<unsigned long long>(function.expected_blocks.load()),
                         static_cast<unsigned long long>(device.callback_count),
                         static_cast<unsigned long long>(device.emitted_count),
                         static_cast<unsigned long long>(device.dropped_count),
                         static_cast<unsigned long long>(function.function_pc),
                         static_cast<unsigned long long>(function.function_size),
                         static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(function.module)),
                         function.name);
        }
        std::free(kernel_snapshot);
    }
    std::fprintf(launches_file,
                 "# format=GOSAN02 host_clock=CLOCK_MONOTONIC launch_id_scope=process\n");
    std::fprintf(launches_file,
                 "launch_id\tgrid_id\thost_timestamp_ns\tcontext_slot\tkernel_slot\tstream\t"
                 "grid_x\tgrid_y\tgrid_z\tblock_x\tblock_y\tblock_z\texpected_blocks\tfunction\n");

    const uint64_t launch_count =
        std::min<uint64_t>(g_tool.launch_records.load(), GO_SAN_MAX_LAUNCHES);
    for (uint64_t i = 0; i < launch_count; ++i) {
        LaunchEntry& launch = g_tool.launches[i];
        if (launch.ready.load(std::memory_order_acquire) != 1)
            continue;
        const FunctionEntry& function = g_tool.functions[launch.kernel_slot];
        std::fprintf(
            launches_file,
            "%llu\t%llu\t%llu\t%u\t%u\t0x%llx\t%u\t%u\t%u\t%u\t%u\t%u\t%llu\t%s\n",
            static_cast<unsigned long long>(launch.launch_id),
            static_cast<unsigned long long>(launch.grid_id),
            static_cast<unsigned long long>(launch.host_timestamp_ns),
            launch.context_slot, launch.kernel_slot,
            static_cast<unsigned long long>(launch.stream),
            launch.grid_x, launch.grid_y, launch.grid_z,
            launch.block_x, launch.block_y, launch.block_z,
            static_cast<unsigned long long>(launch.expected_blocks), function.name);
    }

    std::fprintf(graph_nodes_file,
                 "# format=GOSAN_GRAPH01 observer=compute-sanitizer-graph-node-begin "
                 "host_clock=CLOCK_MONOTONIC identity=(graph_exec,graph_launch_id,node)\n");
    std::fprintf(graph_nodes_file,
                 "graph_exec\tgraph_launch_id\tnode\tgrid_id\thost_timestamp_ns\t"
                 "context_slot\tkernel_slot\tstream\tapi_stream\t"
                 "grid_x\tgrid_y\tgrid_z\tblock_x\tblock_y\tblock_z\t"
                 "expected_blocks\tfunction\n");
    const uint64_t graph_node_count = g_tool.graph_nodes_recorded.load();
    for (uint64_t i = 0; i < graph_node_count; ++i) {
        GraphNodeEntry& graph_node = g_tool.graph_nodes[i];
        if (graph_node.ready.load(std::memory_order_acquire) != 1)
            continue;
        const FunctionEntry& function = g_tool.functions[graph_node.kernel_slot];
        std::fprintf(
            graph_nodes_file,
            "0x%llx\t%u\t0x%llx\t%llu\t%llu\t%u\t%u\t0x%llx\t0x%llx\t"
            "%u\t%u\t%u\t%u\t%u\t%u\t%llu\t%s\n",
            static_cast<unsigned long long>(graph_node.graph_exec),
            graph_node.graph_launch_id,
            static_cast<unsigned long long>(graph_node.node),
            static_cast<unsigned long long>(graph_node.grid_id),
            static_cast<unsigned long long>(graph_node.host_timestamp_ns),
            graph_node.context_slot, graph_node.kernel_slot,
            static_cast<unsigned long long>(graph_node.stream),
            static_cast<unsigned long long>(graph_node.api_stream),
            graph_node.grid_x, graph_node.grid_y, graph_node.grid_z,
            graph_node.block_x, graph_node.block_y, graph_node.block_z,
            static_cast<unsigned long long>(graph_node.expected_blocks),
            function.ready.load(std::memory_order_acquire) == 1
                ? function.name : "<unknown>");
    }

    std::fflush(summary);
    std::fflush(events_file);
    std::fflush(launches_file);
    std::fflush(graph_nodes_file);
    fsync(fileno(summary));
    fsync(fileno(events_file));
    fsync(fileno(launches_file));
    fsync(fileno(graph_nodes_file));
    std::fclose(summary);
    std::fclose(events_file);
    std::fclose(launches_file);
    std::fclose(graph_nodes_file);
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
    if (g_arm_on_first_signal) {
        g_arm_on_first_signal = 0;
        g_arm_requested = 1;
    } else {
        g_flush_requested = 1;
    }
}

bool maybe_arm(ContextEntry& context, Sanitizer_StreamHandle stream)
{
    if (!g_arm_requested)
        return true;
    g_arm_requested = 0;

    GoSanGlobalState initial{};
    initial.events = context.device_events;
    initial.capacity = g_tool.capacity;
    initial.mode = static_cast<uint32_t>(g_tool.mode);
    initial.sample_mask = g_tool.sample_mask;
    const bool success =
        ok(sanitizerMemcpyHostToDeviceAsync(context.device_global, &initial,
                                            sizeof(initial), stream),
           "arm global state") &&
        ok(sanitizerStreamSynchronize(stream), "arm stream synchronize");
    if (!success) {
        g_tool.capture_arm_failures.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    g_tool.attributed_launches.store(0, std::memory_order_relaxed);
    g_tool.launch_records.store(0, std::memory_order_relaxed);
    g_tool.passive_launches_seen.store(0, std::memory_order_relaxed);
    g_tool.launch_table_drops.store(0, std::memory_order_relaxed);
    g_tool.capture_arms.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr, "gpu-observer-sanitizer: capture armed\n");
    return true;
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
    if (!maybe_arm(*context, launch.hStream)) {
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

    if (g_tool.passive_launch_records)
        record_passive_launch(*context, *function, launch, blocks);

    if (function->launches.load(std::memory_order_relaxed) == 1 &&
        !initialize_kernel_state(*context, *function, launch.hStream)) {
        g_tool.callback_data_failures.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (g_tool.function_callback_data) {
        if (function->launches.load(std::memory_order_relaxed) == 1 &&
            !ok(sanitizerSetCallbackData(
                    launch.function,
                    context->device_function_callbacks + function->kernel_slot),
                "sanitizerSetCallbackData"))
            g_tool.callback_data_failures.fetch_add(1, std::memory_order_relaxed);
    } else {
        GoSanCallbackState* callback_state =
            callback_state_for_launch(*context, *function, launch, blocks);
        if (callback_state == nullptr) {
            g_tool.callback_data_failures.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (!ok(sanitizerSetLaunchCallbackData(launch.hLaunch, launch.function,
                                               launch.hStream, callback_state),
                "sanitizerSetLaunchCallbackData"))
            g_tool.callback_data_failures.fetch_add(1, std::memory_order_relaxed);
    }
}

void on_graph_node(Sanitizer_CallbackId cbid,
                   const Sanitizer_GraphNodeLaunchData& graph_node)
{
    if (!g_tool.graph_node_identity ||
        cbid != SANITIZER_CBID_GRAPHS_NODE_LAUNCH_BEGIN ||
        graph_node.isGraphUpload ||
        graph_node.nodeType != CU_GRAPH_NODE_TYPE_KERNEL)
        return;

    const Sanitizer_LaunchData& launch = graph_node.launchData;
    if (launch.functionName == nullptr ||
        std::strstr(launch.functionName, g_tool.target_substring) == nullptr)
        return;

    g_tool.graph_nodes_seen.fetch_add(1, std::memory_order_relaxed);
    ContextEntry* context = find_context(launch.context);
    if (context == nullptr || context->ready.load(std::memory_order_acquire) != 1) {
        g_tool.setup_missed_launches.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    FunctionEntry* function = ensure_function(launch, context_index(context));
    if (function == nullptr) {
        g_tool.setup_missed_launches.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    uint64_t index = g_tool.graph_nodes_recorded.load(std::memory_order_relaxed);
    while (index < GO_SAN_MAX_LAUNCHES &&
           !g_tool.graph_nodes_recorded.compare_exchange_weak(
               index, index + 1, std::memory_order_acq_rel,
               std::memory_order_relaxed)) {
    }
    if (index >= GO_SAN_MAX_LAUNCHES) {
        g_tool.graph_node_table_drops.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    GraphNodeEntry& record = g_tool.graph_nodes[index];
    record.context_slot = context_index(context);
    record.kernel_slot = function->kernel_slot;
    record.graph_launch_id = graph_node.launchId;
    record.grid_id = launch.gridId;
    record.host_timestamp_ns = monotonic_ns();
    record.graph_exec = reinterpret_cast<uintptr_t>(graph_node.graphExec);
    record.node = reinterpret_cast<uintptr_t>(graph_node.node);
    record.stream = reinterpret_cast<uintptr_t>(launch.stream);
    record.api_stream = reinterpret_cast<uintptr_t>(launch.apiStream);
    record.grid_x = launch.gridDim_x;
    record.grid_y = launch.gridDim_y;
    record.grid_z = launch.gridDim_z;
    record.block_x = launch.blockDim_x;
    record.block_y = launch.blockDim_y;
    record.block_z = launch.blockDim_z;
    record.expected_blocks =
        uint64_t(launch.gridDim_x) * launch.gridDim_y * launch.gridDim_z;
    record.ready.store(1, std::memory_order_release);
}

void SANITIZERAPI callback(void*, Sanitizer_CallbackDomain domain,
                           Sanitizer_CallbackId cbid, const void* cbdata)
{
    if (domain == SANITIZER_CB_DOMAIN_RESOURCE &&
        cbid == SANITIZER_CBID_RESOURCE_MODULE_LOADED) {
        on_module_loaded(*static_cast<const Sanitizer_ResourceModuleData*>(cbdata));
    } else if (domain == SANITIZER_CB_DOMAIN_LAUNCH) {
        on_launch(cbid, *static_cast<const Sanitizer_LaunchData*>(cbdata));
    } else if (domain == SANITIZER_CB_DOMAIN_GRAPHS &&
               cbid == SANITIZER_CBID_GRAPHS_NODE_LAUNCH_BEGIN) {
        on_graph_node(cbid,
                      *static_cast<const Sanitizer_GraphNodeLaunchData*>(cbdata));
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
    g_tool.passive_launch_records =
        env_enabled(std::getenv("GPU_OBSERVER_SAN_PASSIVE_LAUNCHES"));
    g_tool.launch_identity =
        env_enabled(std::getenv("GPU_OBSERVER_SAN_LAUNCH_ID"));
    const char* callback_scope =
        std::getenv("GPU_OBSERVER_SAN_CALLBACK_DATA_SCOPE");
    g_tool.function_callback_data =
        callback_scope != nullptr && std::strcmp(callback_scope, "function") == 0;
    g_tool.graph_node_identity =
        env_enabled(std::getenv("GPU_OBSERVER_SAN_GRAPH_NODES"));
    g_arm_on_first_signal =
        (g_tool.launch_identity ||
         env_enabled(std::getenv("GPU_OBSERVER_SAN_ARM_ON_SIGNAL")))
            ? 1
            : 0;

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
    if (g_tool.launch_identity && g_tool.patch_all_modules) {
        std::fprintf(stderr,
                     "gpu-observer-sanitizer: launch identity requires a kernel substring\n");
        return 1;
    }

    if (g_tool.launch_identity && g_tool.passive_launch_records) {
        std::fprintf(stderr,
                     "gpu-observer-sanitizer: launch identity is incompatible with passive launch records\n");
        return 1;
    }

    if (g_tool.launch_identity && g_tool.function_callback_data) {
        std::fprintf(stderr,
                     "gpu-observer-sanitizer: launch identity is incompatible with function callback data\n");
        return 1;
    }

    if (g_tool.graph_node_identity && g_tool.patch_all_modules) {
        std::fprintf(stderr,
                     "gpu-observer-sanitizer: graph node identity requires a kernel substring\n");
        return 1;
    }

    std::signal(SIGUSR2, handle_signal);
    const SanitizerResult subscribed = sanitizerSubscribe(&g_tool.subscriber, callback, nullptr);
    if (!ok(subscribed, "sanitizerSubscribe"))
        return 1;
    ok(sanitizerEnableDomain(1, g_tool.subscriber, SANITIZER_CB_DOMAIN_RESOURCE),
       "enable resource domain");
    ok(sanitizerEnableDomain(1, g_tool.subscriber, SANITIZER_CB_DOMAIN_LAUNCH),
       "enable launch domain");
    if (g_tool.graph_node_identity)
        ok(sanitizerEnableDomain(1, g_tool.subscriber, SANITIZER_CB_DOMAIN_GRAPHS),
           "enable graphs domain");
    if (g_tool.flush_on_sync)
        ok(sanitizerEnableDomain(1, g_tool.subscriber, SANITIZER_CB_DOMAIN_SYNCHRONIZE),
           "enable synchronize domain");
    std::fprintf(stderr,
                 "gpu-observer-sanitizer: initialized mode=%s capacity=%llu sample=1/%u target=%s callback_data_scope=%s launch_identity=%u passive_launch_records=%u graph_node_identity=%u\n",
                 mode_name(g_tool.mode),
                 static_cast<unsigned long long>(g_tool.capacity),
                 g_tool.sample_mask + 1,
                 g_tool.patch_all_modules ? "<all-modules>" : g_tool.target_substring,
                 g_tool.function_callback_data ? "function" : "launch",
                 g_tool.launch_identity ? 1U : 0U,
                 g_tool.passive_launch_records ? 1U : 0U,
                 g_tool.graph_node_identity ? 1U : 0U);
    return 0;
}

} // namespace

int gpu_observer_sanitizer_initializer = initialize();

