/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

// util
#define LANE_COUNT 32
#define LANE_MASK 31
#define LOG_LANE 5
#define WARP_IDX (threadIdx.x >> LOG_LANE)
#define FULL_MASK 0xffffffff

#define ONGOING 0
#define PARTIAL 1
#define FULL 2
#define FLAG_MASK 3

#define N_WARPS 8
#define FLOAT4_PER_THREAD 4

__device__ __forceinline__ u_int32_t getLaneID() {
    u_int32_t laneID;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneID));
    return laneID;
}

__device__ __forceinline__ float inclusiveWarpScan(float val) {
    #pragma unroll
    for(int i = 1; i <= 16; i <<= 1) {
        const float sum = __shfl_up_sync(FULL_MASK, val, i, LANE_COUNT);
        if(getLaneID() >= i) {
            val += sum;
        }
    }

    return val;
}

__device__ __forceinline__ float inclusiveWarpScanCircular(float val) {
    #pragma unroll
    for(int i = 1; i <= 16; i <<= 1) {
        const float sum = __shfl_up_sync(FULL_MASK, val, i, LANE_COUNT);
        if(getLaneID() >= i) {
            val += sum;
        }
    }

    return __shfl_sync(FULL_MASK, val, getLaneID() + LANE_MASK); // LANE_MASK & LANE_MASK
}

__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for(int mask = 16; mask; mask >>= 1) {
        val += __shfl_xor_sync(FULL_MASK, val, mask, LANE_COUNT);
    }

    return val;
}

__device__ __forceinline__ float4 setXAndAddYZW(float add, float4 val) {
    return make_float4(add, val.y + add, val.z + add, val.w + add);
}

__device__ __forceinline__ float4 addFloatToVec4(float add, float4 val) {
    return make_float4(val.x + add, val.y + add, val.z + add, val.w + add);
}

__device__ __forceinline__ float reduceFloat4(float4 val) {
    return val.x + val.y + val.z + val.w;
}

template <u_int32_t PER_THREAD>
__device__ __forceinline__ void scanInclusiveFinal(float4* totalScan, float* scan, float* s_warpReduction, const u_int32_t offset) {
    float warpReduction = 0.f;
    #pragma unroll
    for(u_int32_t i = getLaneID() + offset, j = 0; j < PER_THREAD; i += LANE_COUNT, ++j) {
        totalScan[j] = reinterpret_cast<float4*>(scan)[i];
        totalScan[j].y += totalScan[j].x;
        totalScan[j].z += totalScan[j].y;
        totalScan[j].w += totalScan[j].z;

        const float sum = inclusiveWarpScanCircular(totalScan[j].w);
        totalScan[j] = addFloatToVec4((getLaneID() ? sum: 0) + warpReduction, totalScan[j]);
        warpReduction += __shfl_sync(FULL_MASK, sum, 0);
     }

     if(!getLaneID()) {
        s_warpReduction[WARP_IDX] = warpReduction;
     }
}

template <u_int32_t PER_THREAD>
__device__ __forceinline__ void scanInclusivePartial(float4* totalScan, float* s_warpReduction, const u_int32_t offset, const u_int32_t vecSize) {
    float warpReduction = 0.f;
    #pragma unroll
    for(u_int32_t i = getLaneID() + offset, j = 0; j < PER_THREAD, i += LANE_COUNT, ++j) {
        totalScan[j] = i < vecSize? reinterpret_cast<float4*>(scan)[i]: make_float4(0.f, 0.f, 0.f, 0.f);
        totalScan[j].y += totalScan[j].x;
        totalScan[j].z += totalScan[j].y;
        totalScan[j].w += totalScan[j].z;

        const float sum = inclusiveWarpScanCircular(totalScan[j].w);
        totalScan[j] = addFloatToVec4((getLaneID() ? sum: 0) + warpReduction, totalScan[j]);
        warpReduction += __shfl_sync(FULL_MASK, sum, 0);
    }

     if(!getLaneID()) {
        s_warpReduction[WARP_IDX] = warpReduction;
     }
}

template <u_int32_t PER_THREAD>
__device__ __forceinline__ void PropagateFull(float4* tScan, float* scan, const float prevReduction, const u_int32_t offset) {
    #pragma unroll
    for (uint32_t i = getLaneId() + offset, j = 0; j < PER_THREAD; i += LANE_COUNT, ++j) {
        reinterpret_cast<float4*>(scan)[i] = addFloatToVec4(prevReduction, tScan[j]);
    }
}

template <u_int32_t PER_THREAD>
__device__ __forceinline__ void PropagatePartial(float4* tScan, float* scan, const float prevReduction, const u_int32_t offset, const u_int32_t vecSize) {
    #pragma unroll
    for (uint32_t i = getLaneId() + offset, j = 0; j < PER_THREAD; i += LANE_COUNT, ++j) {
        if (i < vecSize) {
            reinterpret_cast<float4*>(scan)[i] = addFloatToVec4(prevReduction, tScan[j]);
        }
    }
}

__device__ __forceinline__ void lookback(const u_int32_t partIdx, float localReduction, float & s_broadcast, volatile float* blockReduction) {
        float prevReduction = 0.f;
        u_int32_t lookbackIndex = partIdx - 1;
        while (true) {
            const float flagPayload = blockReduction[lookbackIndex];
            if ((flagPayload & FLAG_MASK) > ONGOING) {
                prevReduction += flagPayload >> 2;
                if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE) {
                    s_broadcast = prevReduction;
                    atomicExch((uint32_t*)&threadBlockReduction[partIndex],
                               FLAG_INCLUSIVE | prevReduction + localReduction << 2);
                    break;
                } else {
                    lookbackIndex--;
                }
            }
        }
}

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE
__global__
void preScanKernel(float *out, float *in, unsigned in_size) {
    extern __shared__ float temp[];
    int thid = threadIdx.x;
    int pout = 0, pin = 1;
    temp[pout * in_size + thid] = (thid > 0)? in[thid - 1]: 0;
    __syncthreads();
    for(int offset = 1; offset < in_size; offset += 2) {
        pout = 1 - pout;
        pin = 1 - pout;
        if(thid >= offset) {
            temp[pout * in_size + thid] += temp[pin * in_size + thid - offset];
        } else {
            temp[pout * in_size + thid] = temp[pin * in_size + thid];
        }
        __syncthreads();
    }
    out[thid] = temp[pout * in_size + thid];
}


/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void preScan(float *out, float *in, unsigned in_size) {
    int grid_size = in_size + (BLOCK_SIZE - 1) / BLOCK_SIZE;
    preScanKernel<<<BLOCK_SIZE, grid_size, BLOCK_SIZE>>>(out, in, in_size);
}

