// fast_reduce_ampere.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while(0)

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T v, unsigned mask) {
    return __reduce_add_sync(mask, v);
}

template <typename T>
__global__ void reduce_blocks_sum(const T* __restrict__ x, size_t n, T* __restrict__ partials, int elts_per_thread)
{
    // Vectorized pointer
    const int V = 4; // float4
    const size_t vecN = n / V;
    const float4* __restrict__ xv = reinterpret_cast<const float4*>(x);

    // Grid-stride over vectorized data
    size_t tid   = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride= blockDim.x * gridDim.x;

    float acc = 0.0f;

    // Extra per-thread work to improve arithmetic intensity
    for (int chunk = 0; chunk < elts_per_thread; ++chunk) {
        for (size_t i = tid; i < vecN; i += stride) {
            float4 v = xv[i];
            acc += v.x + v.y + v.z + v.w;
        }
    }

    // Handle tail elements (n not multiple of 4)
    size_t tail_start = vecN * V;
    for (size_t i = tail_start + tid; i < n; i += stride) {
        acc += x[i];
    }

    // Warp-level redux
    unsigned mask = __activemask();
    float warp_sum = warp_reduce_sum(acc, mask);

    // Shared memory for one value per warp
    __shared__ float warp_partials[32];
    int lane   = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    if (lane == 0) warp_partials[warpId] = warp_sum;
    __syncthreads();

    // Reduce warp partials with warp 0
    if (warpId == 0) {
        int numWarps = (blockDim.x + 31) >> 5;
        float val = (lane < numWarps) ? warp_partials[lane] : 0.0f;
        unsigned wmask = __ballot_sync(0xFFFFFFFFu, lane < numWarps);
        float block_sum = __reduce_add_sync(wmask, val);
        if (lane == 0) partials[blockIdx.x] = block_sum;
    }
}

template <typename T>
__global__ void final_reduce_sum(const T* __restrict__ partials, size_t m, T* __restrict__ out)
{
    size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    float acc = 0.0f;
    for (size_t i = tid; i < m; i += stride) acc += partials[i];

    unsigned mask = __activemask();
    float warp_sum = __reduce_add_sync(mask, acc);

    __shared__ float warp_partials[32];
    int lane   = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    if (lane == 0) warp_partials[warpId] = warp_sum;
    __syncthreads();

    if (warpId == 0) {
        int numWarps = (blockDim.x + 31) >> 5;
        float val = (lane < numWarps) ? warp_partials[lane] : 0.0f;
        unsigned wmask = __ballot_sync(0xFFFFFFFFu, lane < numWarps);
        float block_sum = __reduce_add_sync(wmask, (float)val);
        if (lane == 0) *out = block_sum; // single block preferred; or atomicAdd if multiple blocks
    }
}

int main(int argc, char** argv) {
    size_t n = 4'000'000;
    int blocks = 8192;       // large grid to saturate 82 SMs
    int block_size = 256;    // sweet spot for Ampere
    int elts_per_thread = 2; // modest chunking
    if (argc > 1) n = strtoull(argv[1], nullptr, 10);

    // Host init
    std::vector<float> h(n, 1.0f);
    float expected = float(n);

    // Device buffers
    float *d_x = nullptr, *d_partials = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_partials, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

    // Warm-up
    reduce_blocks_sum<float><<<blocks, block_size>>>(d_x, n, d_partials, elts_per_thread);
    final_reduce_sum<float><<<1, 256>>>(d_partials, blocks, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    reduce_blocks_sum<float><<<blocks, block_size>>>(d_x, n, d_partials, elts_per_thread);
    final_reduce_sum<float><<<1, 256>>>(d_partials, blocks, d_out);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    float result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    double bytes = double(n) * sizeof(float);
    double gbps = (bytes / (ms / 1000.0)) / 1e9;
    printf("n=%zu | blocks=%d | block=%d | elts/thread=%d | time=%.3f ms | %.1f GB/s\n",
           n, blocks, block_size, elts_per_thread, ms, gbps);
    printf("Result %.6f (expected %.6f) | rel err=%.3e\n",
           result, expected, fabs(result-expected)/fmaxf(1.0f,fabs(expected)));

    cudaFree(d_x); cudaFree(d_partials); cudaFree(d_out);
    return 0;
}
