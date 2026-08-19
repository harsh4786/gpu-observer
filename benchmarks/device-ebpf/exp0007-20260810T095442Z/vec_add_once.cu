#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < n)
		c[index] = a[index] + b[index];
}

static bool cuda_ok(cudaError_t status, const char *operation)
{
	if (status == cudaSuccess)
		return true;
	std::fprintf(stderr, "%s failed: %s\n", operation,
		     cudaGetErrorString(status));
	return false;
}

int main()
{
	constexpr int element_count = 1 << 12;
	constexpr int threads_per_block = 128;
	const int block_count =
		(element_count + threads_per_block - 1) / threads_per_block;
	const size_t bytes = element_count * sizeof(float);

	std::vector<float> a(element_count), b(element_count), c(element_count);
	for (int i = 0; i < element_count; ++i) {
		a[i] = static_cast<float>(i);
		b[i] = static_cast<float>(3 * i);
	}

	float *device_a = nullptr;
	float *device_b = nullptr;
	float *device_c = nullptr;
	if (!cuda_ok(cudaMalloc(&device_a, bytes), "cudaMalloc(a)") ||
	    !cuda_ok(cudaMalloc(&device_b, bytes), "cudaMalloc(b)") ||
	    !cuda_ok(cudaMalloc(&device_c, bytes), "cudaMalloc(c)"))
		return 2;

	if (!cuda_ok(cudaMemcpy(device_a, a.data(), bytes,
				    cudaMemcpyHostToDevice), "copy(a)") ||
	    !cuda_ok(cudaMemcpy(device_b, b.data(), bytes,
				    cudaMemcpyHostToDevice), "copy(b)"))
		return 3;

	vectorAdd<<<block_count, threads_per_block>>>(device_a, device_b,
						     device_c, element_count);
	if (!cuda_ok(cudaGetLastError(), "vectorAdd launch") ||
	    !cuda_ok(cudaDeviceSynchronize(), "vectorAdd synchronize") ||
	    !cuda_ok(cudaMemcpy(c.data(), device_c, 2 * sizeof(float),
				    cudaMemcpyDeviceToHost), "copy(c)"))
		return 4;

	const bool correct = std::fabs(c[0]) < 0.001f &&
			     std::fabs(c[1] - 4.0f) < 0.001f;
	std::printf("vectorAdd C[0]=%.1f C[1]=%.1f blocks=%d threads=%d status=%s\n",
		    c[0], c[1], block_count, threads_per_block,
		    correct ? "PASS" : "FAIL");

	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);
	return correct ? 0 : 5;
}
