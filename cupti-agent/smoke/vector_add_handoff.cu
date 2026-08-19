#include <cuda_runtime.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#define CUDA_CHECK(operation)                                                   \
    do {                                                                        \
        const cudaError_t status = (operation);                                 \
        if (status != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s failed: %s\n", #operation,                 \
                         cudaGetErrorString(status));                            \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (false)

__global__ void vector_add_handoff(const float* lhs, const float* rhs,
                                   float* output, int count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
        output[index] = lhs[index] + rhs[index];
}

void launch(float* lhs, float* rhs, float* output, int count)
{
    constexpr int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    vector_add_handoff<<<blocks, threads>>>(lhs, rhs, output, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main()
{
    constexpr int count = 4096;
    constexpr size_t bytes = count * sizeof(float);
    float* lhs = nullptr;
    float* rhs = nullptr;
    float* output = nullptr;
    CUDA_CHECK(cudaMalloc(&lhs, bytes));
    CUDA_CHECK(cudaMalloc(&rhs, bytes));
    CUDA_CHECK(cudaMalloc(&output, bytes));
    CUDA_CHECK(cudaMemset(lhs, 0, bytes));
    CUDA_CHECK(cudaMemset(rhs, 0, bytes));

    // The first launch causes module patching; the second proves the patch is
    // active before the subscriber is handed off.
    launch(lhs, rhs, output, count);
    launch(lhs, rhs, output, count);

    if (kill(getpid(), SIGUSR1) != 0)
        return EXIT_FAILURE;
    usleep(500000);

    // These launches must appear in both CUPTI activity intervals and the
    // already-installed SASS block probe after Sanitizer has unsubscribed.
    for (int iteration = 0; iteration < 3; ++iteration)
        launch(lhs, rhs, output, count);

    if (kill(getpid(), SIGUSR2) != 0)
        return EXIT_FAILURE;
    usleep(500000);

    CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFree(rhs));
    CUDA_CHECK(cudaFree(lhs));
    std::puts("vector_add_handoff_ok warmup=2 measured=3");
    return EXIT_SUCCESS;
}
