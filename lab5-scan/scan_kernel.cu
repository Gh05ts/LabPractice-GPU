/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 1024

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE

__device__ __forceinline__ float identity() { return 0.0f; }

// Warp-level inclusive scan
__device__ __forceinline__ float warp_inclusive_scan(float val, unsigned mask = 0xFFFFFFFFu)
{
  const int lane = threadIdx.x & 31;
  for (int offset = 1; offset < 32; offset <<= 1) {
    float n = __shfl_up_sync(mask, val, offset);
    if (lane >= offset) val += n;
  }
  return val;
}

// Block-level inclusive scan (one value per thread)
__device__ float block_inclusive_scan(float val, float* block_total_out)
{
  __shared__ float warp_totals[BLOCK_SIZE / 32];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float scanned = warp_inclusive_scan(val);

  if (lane == 31) warp_totals[warp] = scanned;
  __syncthreads();

  if (warp == 0) {
    float wt = (lane < (BLOCK_SIZE / 32)) ? warp_totals[lane] : identity();
    float wt_scanned = warp_inclusive_scan(wt);
    if (lane < (BLOCK_SIZE / 32)) warp_totals[lane] = wt_scanned;
  }
  __syncthreads();

  float warp_offset = (warp > 0) ? warp_totals[warp - 1] : identity();
  scanned += warp_offset;

  if (block_total_out && threadIdx.x == BLOCK_SIZE - 1) {
    *block_total_out = scanned; // total of this block
  }
  return scanned;
}

// Kernel A: per-block scan, write exclusive per-element results and block totals
__global__ void scan_blocks_kernel_exclusive(const float* __restrict__ d_in,
                                             float* __restrict__ d_out,
                                             float* __restrict__ d_block_totals,
                                             int N)
{
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const bool in_range = (gid < N);
  float val = in_range ? d_in[gid] : identity();

  __shared__ float block_total;
  float inclusive = block_inclusive_scan(val, &block_total);

  if (in_range) {
    // exclusive = inclusive - input
    d_out[gid] = inclusive - val;
  }

  if (threadIdx.x == BLOCK_SIZE - 1) {
    d_block_totals[blockIdx.x] = block_total;
  }
}

// Kernel B: add scanned block offsets (exclusive) to make global exclusive
__global__ void add_block_offsets_kernel_exclusive(float* __restrict__ d_out,
                                                   const float* __restrict__ d_block_totals_scanned_excl,
                                                   int N)
{
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= N) return;

  // For exclusive scan, offset for block b is exclusive_scan(block_totals)[b]
  float offset = d_block_totals_scanned_excl[blockIdx.x];
  d_out[gid] += offset;
}

// Recursive helper: exclusive-scan an array of length M (device arrays)
void exclusive_scan_recursive(float* d_out, const float* d_in, int M)
{
  if (M <= 0) return;

  const int blockSize = BLOCK_SIZE;
  const int numBlocks = (M + blockSize - 1) / blockSize;

  // Single-block fast path
  if (numBlocks == 1) {
    float* d_dummy = nullptr;
    cudaMalloc(&d_dummy, sizeof(float));
    scan_blocks_kernel_exclusive<<<1, blockSize>>>(d_in, d_out, d_dummy, M);
    cudaFree(d_dummy);
    return;
  }

  // General hierarchical path
  float* d_level_out = nullptr;            // per-element exclusive results at this level
  float* d_block_totals = nullptr;         // per-block totals (inclusive sum of each block)
  float* d_block_totals_scanned_excl = nullptr; // exclusive scan of block totals

  cudaMalloc(&d_level_out, sizeof(float) * M);
  cudaMalloc(&d_block_totals, sizeof(float) * numBlocks);
  cudaMalloc(&d_block_totals_scanned_excl, sizeof(float) * numBlocks);

  // Pass 1: per-block exclusive results + collect block totals
  scan_blocks_kernel_exclusive<<<numBlocks, blockSize>>>(d_in, d_level_out, d_block_totals, M);

  // Pass 2: exclusive scan of block totals
  // We can reuse the same routine recursively: out = exclusive_scan(d_block_totals)
  exclusive_scan_recursive(d_block_totals_scanned_excl, d_block_totals, numBlocks);

  // Pass 3: add offsets to each block's elements
  add_block_offsets_kernel_exclusive<<<numBlocks, blockSize>>>(d_level_out, d_block_totals_scanned_excl, M);

  // Copy level result to output
  cudaMemcpy(d_out, d_level_out, sizeof(float) * M, cudaMemcpyDeviceToDevice);

  cudaFree(d_level_out);
  cudaFree(d_block_totals);
  cudaFree(d_block_totals_scanned_excl);
}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void preScan(float *out, float *in, uint in_size) {
    exclusive_scan_recursive(out, in, in_size);
}

// Per-block state published to global memory
// status: 0 = EMPTY, 1 = AGG_READY (total written), 2 = PREFIX_READY (prefix written)
struct BlockState {
    float total;            // aggregate sum of this block's tile
    float prefix;           // exclusive prefix sum of all blocks before this one
    volatile int status;    // volatile to prevent caching across SMs
};

// GONE

// Decoupled look-back exclusive scan kernel
// out[0] = 0, out[i] = sum_{j=0}^{i-1} in[j]
__global__ void decoupled_lookback_exclusive_scan(const float* __restrict__ d_in,
                                                  float* __restrict__ d_out,
                                                  BlockState* __restrict__ d_state,
                                                  int N)
{
    const int b   = blockIdx.x;
    const int tid = threadIdx.x;
    const int gid = b * blockDim.x + tid;
    const bool in_range = (gid < N);

    // 1) Local block scan (inclusive) and convert to exclusive per-element
    float x = in_range ? d_in[gid] : identity();
    __shared__ float block_total;
    float inclusive = block_inclusive_scan(x, &block_total);
    float exclusive = inclusive - x;  // per-element exclusive within the block

    // 2) Publish this block's aggregate total and mark AGG_READY
    if (tid == blockDim.x - 1) {
        d_state[b].total = block_total;
        __threadfence();           // make total visible before status
        d_state[b].status = 1;     // AGG_READY
        __threadfence();           // make status visible
    }
    __syncthreads();

    // 3) Compute this block's prefix via decoupled look-back (robust, no skip pointers)
    __shared__ float block_prefix;
    if (tid == 0) {
        float prefix = 0.0f;

        if (b == 0) {
            // First block: prefix is 0 (exclusive semantics)
            prefix = 0.0f;
        } else {
            int j = b - 1;
            while (true) {
                // Wait until predecessor has at least AGG_READY
                while (d_state[j].status == 0) {
                    // busy wait; relies on forward progress of earlier blocks
                }

                int st = d_state[j].status;
                if (st == 2) {
                    // Predecessor has cached prefix; offset(b) needs prefix(j) + total(j)
                    prefix += d_state[j].prefix + d_state[j].total;
                    break;
                } else { // st == 1 (AGG_READY)
                    // Accumulate this predecessor's total and continue looking back
                    prefix += d_state[j].total;
                    if (j == 0) break; // reached the beginning
                    --j;
                }
            }
        }

        // Publish this block's PREFIX and mark PREFIX_READY
        d_state[b].prefix = prefix;
        __threadfence();           // make prefix visible
        d_state[b].status = 2;     // PREFIX_READY
        __threadfence();           // make status visible

        block_prefix = prefix;
    }
    __syncthreads();

    // 4) Add block prefix to per-element exclusive results to globalize
    if (in_range) {
        d_out[gid] = exclusive + block_prefix;
    }
}

// Host API: decoupled look-back exclusive scan for float
void exclusive_scan_float_dlb(float* d_out, const float* d_in, int N)
{
    const int blockSize = BLOCK_SIZE;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    BlockState* d_state = nullptr;
    cudaMalloc(&d_state, sizeof(BlockState) * numBlocks);
    cudaMemset((void*)d_state, 0, sizeof(BlockState) * numBlocks); // status=0

    decoupled_lookback_exclusive_scan<<<numBlocks, blockSize>>>(d_in, d_out, d_state, N);

    cudaFree(d_state);
}