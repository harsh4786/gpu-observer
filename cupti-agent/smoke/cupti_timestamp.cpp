#include <cupti.h>

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>

namespace {

uint64_t monotonic_ns() {
    timespec value{};
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        std::perror("clock_gettime");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<uint64_t>(value.tv_sec) * 1'000'000'000ULL +
           static_cast<uint64_t>(value.tv_nsec);
}

void check(CUptiResult result, const char* operation) {
    if (result == CUPTI_SUCCESS) {
        return;
    }

    const char* message = "unknown CUPTI error";
    cuptiGetResultString(result, &message);
    std::fprintf(stderr, "%s failed: %s\n", operation, message);
    std::exit(EXIT_FAILURE);
}

struct CalibrationSample {
    int64_t offset_ns;
    uint64_t uncertainty_ns;
};

CalibrationSample sample_clock_pair() {
    uint64_t cupti_ns = 0;
    const uint64_t monotonic_before_ns = monotonic_ns();
    check(cuptiGetTimestamp(&cupti_ns), "cuptiGetTimestamp");
    const uint64_t monotonic_after_ns = monotonic_ns();

    const uint64_t monotonic_midpoint_ns =
        monotonic_before_ns +
        (monotonic_after_ns - monotonic_before_ns) / 2;
    return {
        .offset_ns = static_cast<int64_t>(monotonic_midpoint_ns) -
                     static_cast<int64_t>(cupti_ns),
        .uncertainty_ns = (monotonic_after_ns - monotonic_before_ns) / 2,
    };
}

}  // namespace

int main() {
    constexpr size_t warmup_samples = 8;
    constexpr size_t measured_samples = 64;

    uint32_t runtime_api = 0;
    check(cuptiGetVersion(&runtime_api), "cuptiGetVersion");
    if (runtime_api != CUPTI_API_VERSION) {
        std::fprintf(
            stderr,
            "CUPTI API mismatch: header=%u runtime=%u\n",
            CUPTI_API_VERSION,
            runtime_api);
        return EXIT_FAILURE;
    }

    for (size_t index = 0; index < warmup_samples; ++index) {
        static_cast<void>(sample_clock_pair());
    }

    std::array<CalibrationSample, measured_samples> samples{};
    std::array<uint64_t, measured_samples> uncertainties{};
    CalibrationSample best{
        .offset_ns = 0,
        .uncertainty_ns = std::numeric_limits<uint64_t>::max(),
    };
    int64_t minimum_offset_ns = std::numeric_limits<int64_t>::max();
    int64_t maximum_offset_ns = std::numeric_limits<int64_t>::min();

    for (size_t index = 0; index < measured_samples; ++index) {
        samples[index] = sample_clock_pair();
        uncertainties[index] = samples[index].uncertainty_ns;
        if (samples[index].uncertainty_ns < best.uncertainty_ns) {
            best = samples[index];
        }
        minimum_offset_ns =
            std::min(minimum_offset_ns, samples[index].offset_ns);
        maximum_offset_ns =
            std::max(maximum_offset_ns, samples[index].offset_ns);
    }

    std::sort(uncertainties.begin(), uncertainties.end());
    const uint64_t median_uncertainty_ns =
        uncertainties[measured_samples / 2];
    const uint64_t offset_span_ns =
        static_cast<uint64_t>(maximum_offset_ns - minimum_offset_ns);

    std::printf(
        "header_api=%u runtime_api=%u "
        "samples=%zu best_offset_ns=%" PRId64 " "
        "best_uncertainty_ns=%" PRIu64 " "
        "median_uncertainty_ns=%" PRIu64 " "
        "offset_span_ns=%" PRIu64 "\n",
        CUPTI_API_VERSION,
        runtime_api,
        measured_samples,
        best.offset_ns,
        best.uncertainty_ns,
        median_uncertainty_ns,
        offset_span_ns);

    return EXIT_SUCCESS;
}
