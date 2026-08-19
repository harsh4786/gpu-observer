#pragma once

#include <stdint.h>

// Shared host/device wire types. Keep these POD and version the file header.
// The event allocation is bounded at process startup; callbacks never allocate.

static constexpr uint32_t GO_SAN_FORMAT_VERSION = 1;
static constexpr uint32_t GO_SAN_MAX_CONTEXTS = 4;
static constexpr uint32_t GO_SAN_MAX_FUNCTIONS = 8192;
static constexpr uint32_t GO_SAN_MAX_MODULES = 2048;
static constexpr uint64_t GO_SAN_MAX_EVENTS = 262144;

enum GoSanMode : uint32_t {
    GO_SAN_SUBSCRIBER = 0,
    GO_SAN_BLOCK_NOOP = 1,
    GO_SAN_BLOCK_COUNTER = 2,
    GO_SAN_BLOCK_EVENT = 3,
    GO_SAN_SAMPLED_MEMORY_BARRIER = 4,
    GO_SAN_FULL_MEMORY = 5,
};

enum GoSanEventKind : uint16_t {
    GO_SAN_EVENT_BLOCK = 1,
    GO_SAN_EVENT_GLOBAL_READ = 2,
    GO_SAN_EVENT_GLOBAL_WRITE = 3,
    GO_SAN_EVENT_SHARED_READ = 4,
    GO_SAN_EVENT_SHARED_WRITE = 5,
    GO_SAN_EVENT_LOCAL_READ = 6,
    GO_SAN_EVENT_LOCAL_WRITE = 7,
    GO_SAN_EVENT_BARRIER = 8,
};

struct GoSanEvent {
    uint64_t sequence;
    uint64_t device_timestamp_raw;
    uint64_t pc;
    uint64_t address;
    uint32_t kernel_slot;
    uint32_t block_x;
    uint32_t block_y;
    uint32_t block_z;
    uint32_t thread_linear;
    uint32_t flags;
    uint16_t kind;
    uint16_t access_size;
    uint32_t sm_id;
};

static_assert(sizeof(GoSanEvent) == 64, "event ABI must remain one cache line");

struct GoSanGlobalState {
    GoSanEvent* events;
    uint64_t capacity;
    unsigned long long write_attempts;
    unsigned long long dropped;
    uint32_t mode;
    uint32_t sample_mask;
};

struct GoSanKernelState {
    GoSanGlobalState* global;
    unsigned long long callback_count;
    unsigned long long emitted_count;
    unsigned long long dropped_count;
    uint32_t kernel_slot;
    uint32_t reserved;
};

struct GoSanFileHeader {
    char magic[8];
    uint32_t version;
    uint32_t context_slot;
    uint32_t mode;
    uint32_t event_size;
    uint64_t capacity;
    uint64_t retained_events;
    uint64_t write_attempts;
    uint64_t dropped;
    uint64_t buffer_bytes;
};

