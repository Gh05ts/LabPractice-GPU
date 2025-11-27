#define BLOCK_SIZE 512
#define WARP_SIZE 32
#define LANE_MASK 31
#define LANE_LOG 5

// Each block can have 0, 1, 2 status 
// 0 means computing local sum
// 1 means local sum is computed
// 2 means local sum and all predecessor sums are computed and present in this block
struct BlockState {
    float total;
    float prefix;
    volatile int status;
};

__device__ __forceinline__ float identity() { return 0.0f; }

__device__ __forceinline__ float warp_inclusive_scan(float val, unsigned mask = 0xFFFFFFFFu) {
    const int lane = threadIdx.x & LANE_MASK;
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float n = __shfl_up_sync(mask, val, offset);
        if (lane >= offset) val += n;
    }
    return val;
}

__device__ float block_inclusive_scan(float val, float* block_total_out) {
    __shared__ float warp_totals[BLOCK_SIZE / WARP_SIZE];

    const int lane = threadIdx.x & LANE_MASK;
    const int warp = threadIdx.x >> LANE_LOG;

    float scanned = warp_inclusive_scan(val);

    if (lane == WARP_SIZE - 1) warp_totals[warp] = scanned;
    __syncthreads();

    if (warp == 0) {
        float wt = (lane < (BLOCK_SIZE / WARP_SIZE)) ? warp_totals[lane] : identity();
        float wt_scanned = warp_inclusive_scan(wt);
        if (lane < (BLOCK_SIZE / WARP_SIZE)) warp_totals[lane] = wt_scanned;
    }
    __syncthreads();

    float warp_offset = (warp > 0) ? warp_totals[warp - 1] : identity();
    scanned += warp_offset;

    if (block_total_out && threadIdx.x == BLOCK_SIZE - 1) {
        *block_total_out = scanned;
    }
    return scanned;
}

__global__ void decoupled_lookback_exclusive_scan(const float* __restrict__ d_in, float* __restrict__ d_out, BlockState* __restrict__ d_state, int N) {
    const int bid   = blockIdx.x;
    const int tid = threadIdx.x;
    const int gid = bid * blockDim.x + tid;
    const bool in_range = (gid < N);

    float x = in_range ? d_in[gid] : identity();
    __shared__ float block_total;
    float inclusive = block_inclusive_scan(x, &block_total);
    float exclusive = inclusive - x;

    if (tid == blockDim.x - 1) {
        d_state[bid].total = block_total;
        __threadfence();
        d_state[bid].status = 1;
        __threadfence();
    }
    __syncthreads();

    __shared__ float block_prefix;
    if (tid == 0) {
        float prefix = 0.0f;

        if (bid == 0) {
            // only for exclusive scan
            prefix = 0.0f;
        } else {
            int j = bid - 1;
            while (true) {
                while (d_state[j].status == 0) {
                    // busy wait; relies on forward progress of earlier blocks, guarenteed by cuda runtime
                }

                int st = d_state[j].status;
                // 2 means this block has info for all previous blocks, 1 means only local
                if (st == 2) {
                    prefix += d_state[j].prefix + d_state[j].total;
                    break;
                } else {
                    prefix += d_state[j].total;
                    if (j == 0) break;
                    --j;
                }
            }
        }

        d_state[bid].prefix = prefix;
        __threadfence();
        d_state[bid].status = 2;
        __threadfence();

        block_prefix = prefix;
    }
    __syncthreads();

    if (in_range) {
        d_out[gid] = exclusive + block_prefix;
    }
}

void prescan(float* d_out, const float* d_in, int N)
{
    const int blockSize = BLOCK_SIZE;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    BlockState* d_state = nullptr;
    cudaMalloc(&d_state, sizeof(BlockState) * numBlocks);
    cudaMemset((void*)d_state, 0, sizeof(BlockState) * numBlocks);

    decoupled_lookback_exclusive_scan<<<numBlocks, blockSize>>>(d_in, d_out, d_state, N);

    cudaFree(d_state);
}