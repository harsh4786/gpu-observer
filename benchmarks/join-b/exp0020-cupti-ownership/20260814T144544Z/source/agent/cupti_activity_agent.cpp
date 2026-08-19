#include <cupti.h>
#include <cupti_activity.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr size_t kBufferCount = 8;
constexpr size_t kBufferBytes = 1U << 20;
constexpr size_t kBufferAlignment = 8;
constexpr size_t kTargetBytes = 256;
constexpr size_t kPathBytes = 1024;
constexpr uint64_t kMaxRuntimeRecords = 65536;

struct alignas(kBufferAlignment) ActivityBuffer {
    std::array<uint8_t, kBufferBytes> bytes{};
};

struct ClockPair {
    uint64_t cupti_ns{};
    uint64_t midpoint_ns{};
    uint64_t uncertainty_ns{UINT64_MAX};
};

struct State {
    std::array<ActivityBuffer, kBufferCount> buffers{};
    std::array<std::atomic<uint8_t>, kBufferCount> in_use{};
    std::atomic<uint64_t> buffers_requested{};
    std::atomic<uint64_t> buffers_completed{};
    std::atomic<uint64_t> buffer_exhaustions{};
    std::atomic<uint64_t> records_seen{};
    std::atomic<uint64_t> target_records{};
    std::atomic<uint64_t> invalid_records{};
    std::atomic<uint64_t> dropped_records{};
    std::atomic<uint64_t> runtime_records{};
    std::atomic<uint64_t> runtime_drops{};
    FILE* activities{};
    FILE* runtimes{};
    char control_path[kPathBytes]{};
    char output_prefix[kPathBytes]{};
    char target[kTargetBytes]{};
    ClockPair monotonic{};
    ClockPair monotonic_raw{};
    ClockPair monotonic_end{};
    ClockPair monotonic_raw_end{};
    CUptiResult register_result{CUPTI_ERROR_UNKNOWN};
    CUptiResult enable_result{CUPTI_ERROR_UNKNOWN};
    CUptiResult runtime_enable_result{CUPTI_ERROR_UNKNOWN};
    std::atomic<uint8_t> started{};
    std::atomic<uint8_t> finalized{};
};

State g_state;
int g_control_fd = -1;

uint64_t clock_ns(clockid_t clock)
{
    timespec value{};
    if (clock_gettime(clock, &value) != 0)
        return 0;
    return uint64_t(value.tv_sec) * 1000000000ULL + uint64_t(value.tv_nsec);
}

ClockPair calibrate(clockid_t clock)
{
    ClockPair best{};
    for (unsigned sample = 0; sample < 64; ++sample) {
        const uint64_t before = clock_ns(clock);
        uint64_t cupti = 0;
        if (cuptiGetTimestamp(&cupti) != CUPTI_SUCCESS)
            continue;
        const uint64_t after = clock_ns(clock);
        if (before == 0 || after < before)
            continue;
        const uint64_t uncertainty = after - before;
        if (uncertainty < best.uncertainty_ns)
            best = ClockPair{cupti, before + uncertainty / 2, uncertainty};
    }
    return best;
}

const char* result_name(CUptiResult result)
{
    const char* text = nullptr;
    if (cuptiGetResultString(result, &text) != CUPTI_SUCCESS || text == nullptr)
        return "CUPTI_ERROR_UNAVAILABLE";
    return text;
}

void CUPTIAPI request_buffer(uint8_t** buffer, size_t* size, size_t* max_records)
{
    g_state.buffers_requested.fetch_add(1, std::memory_order_relaxed);
    for (size_t index = 0; index < g_state.buffers.size(); ++index) {
        uint8_t expected = 0;
        if (g_state.in_use[index].compare_exchange_strong(
                expected, 1, std::memory_order_acq_rel)) {
            *buffer = g_state.buffers[index].bytes.data();
            *size = kBufferBytes;
            *max_records = 0;
            return;
        }
    }
    g_state.buffer_exhaustions.fetch_add(1, std::memory_order_relaxed);
    *buffer = nullptr;
    *size = 0;
    *max_records = 0;
}

void release_buffer(uint8_t* buffer)
{
    for (size_t index = 0; index < g_state.buffers.size(); ++index) {
        if (buffer == g_state.buffers[index].bytes.data()) {
            g_state.in_use[index].store(0, std::memory_order_release);
            return;
        }
    }
    g_state.invalid_records.fetch_add(1, std::memory_order_relaxed);
}

bool target_matches(const char* name)
{
    return name != nullptr &&
           (g_state.target[0] == '\0' ||
            std::strstr(name, g_state.target) != nullptr);
}

void write_kernel(const CUpti_ActivityKernel11& kernel)
{
    if (!target_matches(kernel.name))
        return;
    g_state.target_records.fetch_add(1, std::memory_order_relaxed);
    flockfile(g_state.activities);
    const int written = std::fprintf(
        g_state.activities,
        "%" PRIu64 "\t%" PRIu64 "\t%u\t%u\t%u\t%u\t%" PRId64
        "\t%" PRIu64 "\t%u\t%d\t%d\t%d\t%d\t%d\t%d\t%u\t%u\t%s\n",
        kernel.start, kernel.end, kernel.deviceId, kernel.contextId,
        kernel.streamId, kernel.correlationId, kernel.gridId,
        kernel.graphNodeId, kernel.graphId, kernel.gridX, kernel.gridY,
        kernel.gridZ, kernel.blockX, kernel.blockY, kernel.blockZ,
        kernel.channelID, unsigned(kernel.channelType), kernel.name);
    funlockfile(g_state.activities);
    if (written < 0)
        g_state.dropped_records.fetch_add(1, std::memory_order_relaxed);
}
void write_runtime(const CUpti_ActivityAPI& api)
{
    const uint64_t index =
        g_state.runtime_records.fetch_add(1, std::memory_order_relaxed);
    if (index >= kMaxRuntimeRecords) {
        g_state.runtime_drops.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    flockfile(g_state.runtimes);
    const int written = std::fprintf(
        g_state.runtimes,
        "%" PRIu64 "\t%" PRIu64 "\t%u\t%u\t%u\t%u\t%u\n",
        api.start, api.end, api.processId, api.threadId, api.correlationId,
        unsigned(api.cbid), api.returnValue);
    funlockfile(g_state.runtimes);
    if (written < 0)
        g_state.runtime_drops.fetch_add(1, std::memory_order_relaxed);
}

void CUPTIAPI complete_buffer(CUcontext, uint32_t, uint8_t* buffer,
                              size_t, size_t valid_size)
{
    g_state.buffers_completed.fetch_add(1, std::memory_order_relaxed);
    if (buffer == nullptr) {
        g_state.invalid_records.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    CUpti_Activity* record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, valid_size, &record) == CUPTI_SUCCESS) {
        g_state.records_seen.fetch_add(1, std::memory_order_relaxed);
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
            record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            write_kernel(*reinterpret_cast<const CUpti_ActivityKernel11*>(record));
        } else if (record->kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
            write_runtime(*reinterpret_cast<const CUpti_ActivityAPI*>(record));
        }
    }
    release_buffer(buffer);
}

void write_summary()
{
    char path[kPathBytes + 32]{};
    std::snprintf(path, sizeof(path), "%s.summary.tsv", g_state.output_prefix);
    FILE* output = std::fopen(path, "w");
    if (output == nullptr)
        return;
    std::fprintf(
        output,
        "# format=GOCUPTI01 activity_clock=CUPTI_TIMESTAMP_NS "
        "buffer_count=%zu buffer_bytes=%zu total_buffer_bytes=%zu\n",
        kBufferCount, kBufferBytes, kBufferCount * kBufferBytes);
    std::fprintf(
        output,
        "register_result\tenable_result\truntime_enable_result\tbuffers_requested\tbuffers_completed\t"
        "buffer_exhaustions\trecords_seen\ttarget_records\tinvalid_records\t"
        "dropped_records\truntime_records\truntime_drops\n");
    std::fprintf(
        output,
        "%d\t%d\t%d\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64
        "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\n",
        int(g_state.register_result), int(g_state.enable_result),
        int(g_state.runtime_enable_result),
        g_state.buffers_requested.load(std::memory_order_relaxed),
        g_state.buffers_completed.load(std::memory_order_relaxed),
        g_state.buffer_exhaustions.load(std::memory_order_relaxed),
        g_state.records_seen.load(std::memory_order_relaxed),
        g_state.target_records.load(std::memory_order_relaxed),
        g_state.invalid_records.load(std::memory_order_relaxed),
        g_state.dropped_records.load(std::memory_order_relaxed),
        g_state.runtime_records.load(std::memory_order_relaxed),
        g_state.runtime_drops.load(std::memory_order_relaxed));
    std::fprintf(
        output,
        "clock\tcupti_ns\tmidpoint_ns\tuncertainty_ns\toffset_ns\n");
    const auto emit_clock = [&](const char* name, const ClockPair& pair) {
        const int64_t offset =
            int64_t(pair.midpoint_ns) - int64_t(pair.cupti_ns);
        std::fprintf(
            output,
            "%s\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRId64 "\n",
            name, pair.cupti_ns, pair.midpoint_ns, pair.uncertainty_ns, offset);
    };
    emit_clock("CLOCK_MONOTONIC_START", g_state.monotonic);
    emit_clock("CLOCK_MONOTONIC_END", g_state.monotonic_end);
    emit_clock("CLOCK_MONOTONIC_RAW_START", g_state.monotonic_raw);
    emit_clock("CLOCK_MONOTONIC_RAW_END", g_state.monotonic_raw_end);
    std::fclose(output);
}

bool open_outputs()
{
    char path[kPathBytes + 32]{};
    std::snprintf(path, sizeof(path), "%s.activities.tsv",
                  g_state.output_prefix);
    g_state.activities = std::fopen(path, "w");
    if (g_state.activities == nullptr) {
        std::fprintf(stderr, "gpu-observer-cupti: open %s failed: %s\n",
                     path, std::strerror(errno));
        return false;
    }
    std::snprintf(path, sizeof(path), "%s.runtime.tsv",
                  g_state.output_prefix);
    g_state.runtimes = std::fopen(path, "w");
    if (g_state.runtimes == nullptr) {
        std::fprintf(stderr, "gpu-observer-cupti: open %s failed: %s\n",
                     path, std::strerror(errno));
        std::fclose(g_state.activities);
        g_state.activities = nullptr;
        return false;
    }
    std::setvbuf(g_state.activities, nullptr, _IOFBF, 1U << 20);
    std::setvbuf(g_state.runtimes, nullptr, _IOFBF, 1U << 20);
    std::fprintf(
        g_state.activities,
        "# format=GOCUPTI01 observer=cupti-concurrent-kernel "
        "activity_clock=CUPTI_TIMESTAMP_NS target=%s\n",
        g_state.target[0] == '\0' ? "<all>" : g_state.target);
    std::fprintf(
        g_state.activities,
        "start_ns\tend_ns\tdevice\tcontext\tstream\tcorrelation\tgrid_id\t"
        "graph_node_id\tgraph_id\tgrid_x\tgrid_y\tgrid_z\tblock_x\tblock_y\t"
        "block_z\tchannel_id\tchannel_type\tname\n");
    std::fprintf(
        g_state.runtimes,
        "# format=GOCUPTI_RUNTIME01 observer=cupti-runtime-api "
        "activity_clock=CUPTI_TIMESTAMP_NS max_records=%" PRIu64 "\n",
        kMaxRuntimeRecords);
    std::fprintf(
        g_state.runtimes,
        "start_ns\tend_ns\tprocess\tthread\tcorrelation\tcbid\treturn_value\n");
    return true;
}

int start_collection()
{
    uint8_t expected = 0;
    if (!open_outputs()) {
        g_state.started.store(0, std::memory_order_release);
        return 1;
    }
    if (!g_state.started.compare_exchange_strong(expected, 1))
        return g_state.enable_result == CUPTI_SUCCESS ? 0 : 1;
    g_state.monotonic = calibrate(CLOCK_MONOTONIC);
    g_state.monotonic_raw = calibrate(CLOCK_MONOTONIC_RAW);
    g_state.register_result =
        cuptiActivityRegisterCallbacks(request_buffer, complete_buffer);
    if (g_state.register_result == CUPTI_SUCCESS) {
        g_state.enable_result =
            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        if (g_state.enable_result == CUPTI_SUCCESS) {
            g_state.runtime_enable_result =
                cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
        }
    }
    std::fprintf(
        stderr,
        "gpu-observer-cupti: register=%s kernel=%s runtime=%s target=%s "
        "bounded_buffer_bytes=%zu\n",
        result_name(g_state.register_result), result_name(g_state.enable_result),
        result_name(g_state.runtime_enable_result),
        g_state.target[0] == '\0' ? "<all>" : g_state.target,
        kBufferCount * kBufferBytes);
    return (g_state.register_result == CUPTI_SUCCESS &&
            g_state.enable_result == CUPTI_SUCCESS &&
            g_state.runtime_enable_result == CUPTI_SUCCESS)
               ? 0
               : 1;
}

int flush_collection()
{
    if (g_state.enable_result != CUPTI_SUCCESS)
        return 1;
    const CUptiResult result =
        cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    g_state.monotonic_end = calibrate(CLOCK_MONOTONIC);
    g_state.monotonic_raw_end = calibrate(CLOCK_MONOTONIC_RAW);
    if (g_state.activities != nullptr)
        std::fflush(g_state.activities);
    if (g_state.runtimes != nullptr)
        std::fflush(g_state.runtimes);
    write_summary();
    return result == CUPTI_SUCCESS ? 0 : 1;
}
void handle_control_signal(int signal)
{
    const char command = signal == SIGUSR1 ? 'S' : 'F';
    if (g_control_fd >= 0) {
        const ssize_t ignored = write(g_control_fd, &command, 1);
        (void)ignored;
    }
}

void* control_worker(void*)
{
    for (;;) {
        char command = 0;
        const ssize_t count = read(g_control_fd, &command, 1);
        if (count < 0 && errno == EINTR)
            continue;
        if (count <= 0)
            return nullptr;
        const int status =
            command == 'S' ? start_collection()
                           : command == 'F' ? flush_collection() : 1;
        std::fprintf(stderr,
                     "gpu-observer-cupti: control command=%c status=%d\n",
                     command, status);
    }
}

bool start_control_worker()
{
    std::snprintf(g_state.control_path, sizeof(g_state.control_path),
                  "/tmp/gpu-observer-cupti-%ld.fifo", long(getpid()));
    unlink(g_state.control_path);
    if (mkfifo(g_state.control_path, 0600) != 0)
        return false;
    g_control_fd = open(g_state.control_path, O_RDWR | O_CLOEXEC);
    if (g_control_fd < 0) {
        unlink(g_state.control_path);
        g_state.control_path[0] = '\0';
        return false;
    }
    pthread_t thread{};
    if (pthread_create(&thread, nullptr, control_worker, nullptr) != 0) {
        close(g_control_fd);
        g_control_fd = -1;
        unlink(g_state.control_path);
        g_state.control_path[0] = '\0';
        return false;
    }
    pthread_detach(thread);
    std::signal(SIGUSR1, handle_control_signal);
    std::signal(SIGUSR2, handle_control_signal);
    return true;
}

void finalize()
{
    uint8_t expected = 0;
    if (!g_state.finalized.compare_exchange_strong(expected, 1))
        return;
    if (g_state.started.load(std::memory_order_acquire) == 0) {
        if (g_state.control_path[0] != '\0')
            unlink(g_state.control_path);
        return;
    }
    if (g_state.enable_result == CUPTI_SUCCESS)
        cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    if (g_state.monotonic_end.uncertainty_ns == UINT64_MAX) {
        g_state.monotonic_end = calibrate(CLOCK_MONOTONIC);
        g_state.monotonic_raw_end = calibrate(CLOCK_MONOTONIC_RAW);
    }
    if (g_state.activities != nullptr) {
        std::fflush(g_state.activities);
        std::fclose(g_state.activities);
        g_state.activities = nullptr;
    }
    if (g_state.runtimes != nullptr) {
        std::fflush(g_state.runtimes);
        std::fclose(g_state.runtimes);
        g_state.runtimes = nullptr;
    }
    write_summary();
    if (g_state.control_path[0] != '\0')
        unlink(g_state.control_path);
}

void set_output_prefix(const char* configured)
{
    const char* source =
        configured != nullptr ? configured : "/tmp/gpu-observer-cupti";
    const char* marker = std::strstr(source, "%p");
    if (marker == nullptr) {
        std::snprintf(g_state.output_prefix, sizeof(g_state.output_prefix),
                      "%s", source);
        return;
    }
    const int prefix_bytes = int(marker - source);
    std::snprintf(g_state.output_prefix, sizeof(g_state.output_prefix),
                  "%.*s%ld%s", prefix_bytes, source, long(getpid()),
                  marker + 2);
}
int initialize()
{
    const char* prefix =
        std::getenv("GPU_OBSERVER_CUPTI_OUTPUT_PREFIX");
    const char* target =
        std::getenv("GPU_OBSERVER_CUPTI_KERNEL_SUBSTRING");
    set_output_prefix(prefix);
    if (target != nullptr)
        std::snprintf(g_state.target, sizeof(g_state.target), "%s", target);
    std::atexit(finalize);
    const char* deferred = std::getenv("GPU_OBSERVER_CUPTI_DEFER");
    if (deferred != nullptr && std::strcmp(deferred, "1") == 0) {
        if (!start_control_worker()) {
            std::fprintf(stderr, "gpu-observer-cupti: control worker failed\n");
            return 1;
        }
        return 0;
    }
    return start_collection();
}

} // namespace

extern "C" int gpu_observer_cupti_activity_start()
{
    return start_collection();
}
extern "C" int gpu_observer_cupti_activity_flush()
{
    return flush_collection();
}

int gpu_observer_cupti_activity_initializer = initialize();
