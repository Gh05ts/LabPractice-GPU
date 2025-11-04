#include <iostream>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include "support.h"


#define FULL_MASK 0xFFFFFFFF
#define WARP_SIZE 32

__device__ __forceinline__ uint32_t lane_id(uint32_t tid){
    // https://stackoverflow.com/q/44337309
    return tid & (32 - 1);
}

template <typename T>
__device__ T warp_sum(T value, uint32_t mask, uint32_t length){
#pragma unroll
    for (uint32_t i = length / 2; 0 < i; i /= 2){
        T value_tmp = __shfl_down_sync(mask, value, i);
        value += value_tmp;
    }

    return value;
}

template <typename T>
__device__ T block_sum(T value, uint32_t tid, uint32_t n_threads, T shared_memory[]){
    value = warp_sum(value, FULL_MASK, WARP_SIZE);

    if (lane_id(tid) == 0){
        shared_memory[tid / WARP_SIZE] = value;
    }

    __syncthreads();

    uint32_t n_warps = n_threads / WARP_SIZE;

    if (tid < n_warps){
        value = shared_memory[tid];

        // Create a mask for the active threads
        uint32_t mask = (1 << n_warps) - 1;

#pragma unroll
        for (uint32_t i = n_warps / 2; 0 < i; i /= 2){
            T value_tmp = __shfl_down_sync(mask, value, i);
            value += value_tmp;
        }
    }

    return value;
}

template <uint32_t block_size>
__global__ void grid_stride_reduce(uint32_t *array, uint32_t *tmp_array, uint32_t length){
    extern __shared__ uint32_t shared_memory[];

    uint32_t thread_index = threadIdx.x;
    uint32_t global_index = blockIdx.x * (block_size * 2) + thread_index;
    uint32_t grid_size = block_size * 2 * gridDim.x;

    uint32_t value = 0;
    while (global_index < length){
        value += array[global_index] + array[global_index + block_size];
        global_index += grid_size;
    }

    value = block_sum(value, thread_index, block_size, shared_memory);

    if (thread_index == 0){
        tmp_array[blockIdx.x] = value;
    }
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid)
{
    if (blockSize >= 64)
        sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32)
        sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16)
        sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)
        sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)
        sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)
        sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata, unsigned int n)
{
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;
    while (i < n)
    {
        sdata[tid] += g_idata[i] + g_idata[i + blockSize];
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (tid < 64)
        {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32)
        warpReduce<blockSize>(sdata, tid);
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

int32_t main(){
    Timer timer;
    uint32_t length = 1 << 22;//96 * 1024 * 1024;
    uint32_t *host_array = (uint32_t *)malloc(length * sizeof(uint32_t));
    if (!host_array) {
        fprintf(stderr, "Failed to allocate host_array\n");
        return 1;
    }

    for (int i = 0; i < length; ++i){
        host_array[i] = 1;
    }

    uint32_t *array;
    cudaMalloc(&array, length * sizeof(uint32_t));
    cudaMemcpy(array, host_array, length * sizeof(uint32_t), cudaMemcpyHostToDevice);

    int32_t devId = 0;
    int32_t n_sm;
    cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, devId);

    uint32_t grid_size = 32 * n_sm;
    const uint32_t block_size = 256;
    const uint32_t shared_memory = (block_size / WARP_SIZE) * sizeof(int32_t);

    uint32_t *tmp_array;
    cudaMalloc(&tmp_array, grid_size * sizeof(uint32_t));

    startTime(&timer);
    grid_stride_reduce<block_size><<<grid_size, block_size, shared_memory>>>(array, tmp_array, length);
    // grid_stride_reduce<block_size><<<1, block_size, shared_memory>>>(tmp_array, tmp_array, grid_size);
    stopTime(&timer);
    printf("%f s\n", elapsedTime(timer));

    uint32_t result;
    cudaMemcpy(&result, tmp_array, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::cout << "Result: " << result << std::endl;

    startTime(&timer);

    reduce6<block_size><<<grid_size, block_size, shared_memory>>>((int *) array, (int *) tmp_array, length);
    // reduce6<block_size><<<1, block_size, shared_memory>>>((int *) tmp_array, (int *) tmp_array, grid_size);

    stopTime(&timer);
    printf("%f s\n", elapsedTime(timer));


    cudaMemcpy(&result, tmp_array, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::cout << "Result: " << result << std::endl;

    cudaFree(array);
    cudaFree(tmp_array);
    return 0;
}