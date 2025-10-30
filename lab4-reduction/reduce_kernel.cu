/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512
#define SIMPLE

__device__ void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduction(float *out, float *in, unsigned size) {
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    __shared__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float sum = 0.0f;
    if (i < size) sum += in[i];
    if (i + blockDim.x < size) sum += in[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    // for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    //     if (tid < s) {
    //         sdata[tid] += sdata[tid + s];
    //     }
    //     __syncthreads();
    // }

    if (tid < 256) { 
        sdata[tid] += sdata[tid + 256]; 
    } 
    __syncthreads(); 
    // if (blockSize >= 512) {
    // }

    if (tid < 128) { 
        sdata[tid] += sdata[tid + 128]; 
    } 
    __syncthreads();
    // if (blockSize >= 256) {
    // }
    if (tid < 64) { 
        sdata[tid] += sdata[tid + 64]; 
    } 
    __syncthreads(); 
    // if (blockSize >= 128) {
    // }


    // Write result for this block
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) out[blockIdx.x] = sdata[0];
}
