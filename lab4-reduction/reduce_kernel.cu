/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 256
#define SIMPLE
#include <cooperative_groups.h>
// #define N 2
// #define OUTPUTS 1u<<N

namespace cg = cooperative_groups;

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
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;

    // float sum = 0.0f;
    // if (i < size) sum += in[i];
    // if (i + blockDim.x < size) sum += in[i + blockDim.x];
    // sdata[tid] = sum;
    // __syncthreads();
    sdata[tid] = 0;
    while (i < size) {
        sdata[tid] += in[i] + in[i+blockSize];
        i += gridSize;
    }
    __syncthreads();
    // Reduction in shared memory
    // for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    //     if (tid < s) {
    //         sdata[tid] += sdata[tid + s];
    //     }
    //     __syncthreads();
    // }

    // if (tid < 512) { 
    //     sdata[tid] += sdata[tid + 512]; 
    // }
    // __syncthreads(); 

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

__inline __device__
float warpReduceSum(float val) {
    unsigned mask = __activemask();
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__inline__ __device__
float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane]: 0;
    if(wid == 0) val = warpReduceSum(val);
    return val;
}

__global__ 
void deviceReduceKernel(float* in, float* out, int size) {
    float sum = 0.f;
    // #pragma unroll
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    sum = blockReduceSum(sum);
    if(threadIdx.x == 0)
        out[blockIdx.x] = sum;
}

__global__
void deviceReduceWarpAtomicKernel(float* in, float* out, int size) {
    float sum = 0.f;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    sum = warpReduceSum(sum);
    if ((threadIdx.x & (warpSize - 1)) == 0)
        atomicAdd(out, sum);
}

template<int TPB>
__device__ __forceinline__ float warp_reduce_shfl_cg(cg::thread_block_tile<32> tile, float val)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += tile.shfl_down(val, offset);
    }
    return val;
}

template<int TPB>
__device__ __forceinline__ float tile_reduce(cg::thread_block &b, float val, float *shared)
{
    // Partition the block into warp-sized tiles (32)
    auto tile32 = cg::tiled_partition<32>(b);

    // each tile does a warp-level shuffle reduction
    float warp_sum = warp_reduce_shfl_cg<TPB>(tile32, val);

    // lane 0 in each warp writes partial to shared
    unsigned lane = tile32.thread_rank();
    unsigned warp_id = b.thread_rank() / 32;
    if (lane == 0) shared[warp_id] = warp_sum;

    b.sync(); // wait for all warps to write partials

    // first warp reduces the per-warp partials
    float block_sum = 0.0f;
    int nWarps = (b.size() + 31) / 32;
    if (b.thread_rank() < nWarps) block_sum = shared[b.thread_rank()];

    auto first_warp = cg::tiled_partition<32>(b); // static tile for first warp
    if (warp_id == 0) {
        block_sum = warp_reduce_shfl_cg<TPB>(first_warp, block_sum);
    }
    // result is valid in thread 0 of block (warp_id==0 && lane==0)
    return block_sum;
}

// Kernel: each tile (warp) reduces and one thread does atomicAdd
template<int TPB>
__global__ void coop_reduce_kernel(const float * __restrict__ in, float *out, int N)
{
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ float sdata[]; // size = sizeof(float) * ((TPB+31)/32)

    // Per-thread strided accumulation
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;

    // Use vectorized loads when possible: try to read float4 aligned chunks
    // Conservative safe vectorization: do scalar loads here to match blog style;
    // if input is 16B-aligned and N large, replacing the loop with float4 loads is fine.
    for (size_t i = tid; i < (size_t)N; i += stride) {
        sum += __ldg(in + i);
    }

    // Reduce within block using tile/wrap reduction
    float block_sum = tile_reduce<TPB>(cta, sum, sdata);

    // If this is the first lane of a warp and that lane is also the warp-id==0 lane of a block,
    // tile_reduce returned valid final sum in warp 0 lane 0. But we want one thread per tile to do the atomic:
    // choose the lane that is tile leader (tile32.thread_rank() == 0) to aggregate upward into global sum.
    // Simpler: let each tile elect its lane-0 to perform atomicAdd of that tile's partial.
    // We already have per-warp partials in shared; but easier: reuse tile_reduce for per-block full sum and
    // then let block thread 0 perform atomicAdd. However per-blog higher concurrency uses per-tile atomicAdd:
    // We'll implement per-tile atomicAdd: each tile's leader does one atomicAdd of tile's warp_sum.

    // Recompute tile warp_sum (so any tile leader can atomicAdd); this avoids additional global memory ops:
    // (An alternate efficient approach writes per-tile partials to shared and only warp 0 reduces - then
    // leaders of each tile do atomicAdd with their tile partials; we implement leaders -> atomicAdd here.)

    // Partition again to determine tile leader
    auto tile32 = cg::tiled_partition<32>(cta);
    // Each thread computes its warp-level sum again (cheap); the leader will atomicAdd
    float warp_sum = warp_reduce_shfl_cg<TPB>(tile32, sum);

    if (tile32.thread_rank() == 0) {
        // The tile leader performs an atomic add of its warp partial
        atomicAdd(out, warp_sum);
    }
}

// Host helper to run kernel and get final result
bool run_coop_reduce(const float *d_in, float *d_out, int N,
                     int threads = 256, int blocks = 0)
{
    if (blocks == 0) {
        blocks = (N + threads - 1) / threads;
        if (blocks == 0) blocks = 1;
        // optionally cap blocks to a large but reasonable value:
        int maxBlocks = 65535;
        if (blocks > maxBlocks) blocks = maxBlocks;
    }

    // zero output on device
    cudaMemset(d_out, 0, sizeof(float));

    // shared mem size: one float per warp in a block
    size_t shared_bytes = sizeof(float) * ((threads + 31) / 32);

    // Launch with template TPB = threads
    switch (threads) {
        case 128:
            coop_reduce_kernel<128><<<blocks, 128, shared_bytes>>>(d_in, d_out, N);
            break;
        case 256:
            coop_reduce_kernel<256><<<blocks, 256, shared_bytes>>>(d_in, d_out, N);
            break;
        case 512:
            coop_reduce_kernel<512><<<blocks, 512, shared_bytes>>>(d_in, d_out, N);
            break;
        default:
            // fallback: compile-time template must match, choose 256 as default
            coop_reduce_kernel<256><<<blocks, 256, shared_bytes>>>(d_in, d_out, N);
            break;
    }
    return true;
}

__device__ float thread_sum(float *input, int n) {
    float sum = 0.f;

    for(int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < n / 4; 
        i += blockDim.x * gridDim.x)
    {
        float4 in = ((float4*)input)[i];
        sum += in.x + in.y + in.z + in.w;
    }
    return sum;
}

template<int tile_sz>
__device__ float reduce_sum_tile_shfl(thread_block_tile<tile_sz> g, float val) {
    for (int i = g.size() / 2; i > 0; i /= 2) {
        val += g.shfl_down(val, i);
    }
    return val;
}

template<int tile_sz>
__global__ void sum_kernel_tile_shfl(float *sum, float *input, int n) {
    float my_sum = thread_sum(input, n);

    auto tile = tiled_partition<tile_sz>(this_thread_block());
    float tile_sum = reduce_sum_tile_shfl<tile_sz>(tile, my_sum);

    if (tile.thread_rank() == 0) atomicAdd(sum, tile_sum);
}