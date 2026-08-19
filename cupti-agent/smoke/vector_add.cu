#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(operation)                                                   \
    do {                                                                        \
        const cudaError_t status = (operation);                                 \
        if (status != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s failed: %s\n", #operation,               \
                         cudaGetErrorString(status));                           \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (false)

__global__ void vector_add(const float* lhs, const float* rhs, float* output,
                           int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = lhs[index] + rhs[index];
    }
}

int main() {
    constexpr int count = 4096;
    constexpr int threads = 256;
    constexpr int blocks = (count + threads - 1) / threads;
    constexpr size_t bytes = count * sizeof(float);

    float* lhs = nullptr;
    float* rhs = nullptr;
    float* output = nullptr;
    CUDA_CHECK(cudaMalloc(&lhs, bytes));
    CUDA_CHECK(cudaMalloc(&rhs, bytes));
    CUDA_CHECK(cudaMalloc(&output, bytes));
    CUDA_CHECK(cudaMemset(lhs, 0, bytes));
    CUDA_CHECK(cudaMemset(rhs, 0, bytes));

    for (int iteration = 0; iteration < 3; ++iteration) {
        vector_add<<<blocks, threads>>>(lhs, rhs, output, count);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFree(rhs));
    CUDA_CHECK(cudaFree(lhs));
    std::puts("vector_add_ok launches=3");
    return EXIT_SUCCESS;
}
