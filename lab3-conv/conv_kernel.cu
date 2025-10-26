/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include "support.h"
// #include "stdio.h"

__constant__ float M_c[FILTER_SIZE][FILTER_SIZE];
#define OUTPT 4
#define TILE_OUT_DIM (BLOCK_SIZE * OUTPT)
#define S_WIDTH (TILE_OUT_DIM + 2*FILTER_RAD)
#define S_HEIGHT (TILE_OUT_DIM + 2*FILTER_RAD)
#define P_DIM (OUTPT + 2*FILTER_RAD)

using Vec4 = float4;

// #define OUTPT 2
// constexpr int filter_rad = (FILTER_SIZE - 1) / 2;
// constexpr int PATCH = OUTPT + 2*filter_rad;

__global__ 
void convolutionTex(cudaTextureObject_t N, Matrix P) {
    /********************************************************************
    Determine input and output indexes of each thread
    Load a tile of the input image to shared memory
    Apply the filter on the input image tile
    Write the compute values to the output image at the correct indexes
    ********************************************************************/

    constexpr const int filter_rad = (FILTER_SIZE - 1) / 2;

    // INSERT KERNEL CODE HERE
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    if (outCol >= P.width || outRow >= P.height) return;

    float Psum = 0.0f;
    for (int fy = -filter_rad; fy <= filter_rad; fy++) {
        for (int fx = -filter_rad; fx <= filter_rad; fx++) {
            int inpX = outCol + fx;
            int inpY = fy + outRow;
            int filY = fy + filter_rad;
            int filX = fx + filter_rad;
            float pixel = tex2D<float>(N, inpX, inpY);
            float weight = M_c[filY][filX];
            Psum += pixel * weight;
        }
    }    
    P.elements[outRow * P.width + outCol] = Psum;
}

__global__ 
void convolution(Matrix N, Matrix P) {
    /********************************************************************
    Determine input and output indexes of each thread
    Load a tile of the input image to shared memory
    Apply the filter on the input image tile
    Write the compute values to the output image at the correct indexes
    ********************************************************************/

    // constexpr const int filter_rad = (FILTER_SIZE - 1) / 2;
    // printf("TILE SIZE: %d", TILE_SIZE);
    __shared__ float N_Sh[TILE_SIZE][TILE_SIZE];

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int srow = row - FILTER_RAD - threadIdx.y;
    int scol = col - FILTER_RAD - threadIdx.x;

    for(int i = threadIdx.y; i < TILE_SIZE; i += blockDim.y) {
        for(int j = threadIdx.x; j < TILE_SIZE; j += blockDim.x) {
            int ssrow = srow + i;
            int sscol = scol + j;
            
            if(ssrow >= 0 && ssrow < N.height && sscol >= 0 && sscol < N.width) {
                N_Sh[i][j] = N.elements[ssrow * N.width + sscol];
            } else {
                N_Sh[i][j] = 0.0f;
            }
        }
    }
    __syncthreads();

    if(row < N.height && col < N.width) {
        float acc = 0.0f;
        int sx = threadIdx.x + FILTER_RAD;
        int sy = threadIdx.y + FILTER_RAD;

        // #pragma unroll
        for(int ky = -FILTER_RAD; ky <= FILTER_RAD; ky++) {
            for(int kx = -FILTER_RAD; kx <= FILTER_RAD; kx++) {
                float pix = N_Sh[sy + ky][sx + kx];
                float w = M_c[ky + FILTER_RAD][kx + FILTER_RAD];
                acc += pix * w;
            }
        }
        P.elements[row * N.width + col] = acc;
    }
}

__global__
void convolution_tiled_per_thread(Matrix N, Matrix P) {
    // Shared tile sized S_HEIGHT x S_WIDTH
    __shared__ float N_Sh[S_HEIGHT][S_WIDTH];

    // block origin in output coordinate space (top-left output pixel for this block)
    const int outBlockRow = blockIdx.y * TILE_OUT_DIM;
    const int outBlockCol = blockIdx.x * TILE_OUT_DIM;

    // thread coordinates inside block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int threadsPerBlockX = blockDim.x;
    const int threadsPerBlockY = blockDim.y;
    const int tid = ty * threadsPerBlockX + tx;
    const int nThreads = threadsPerBlockX * threadsPerBlockY;

    // global coordinates of the first output this thread will produce
    const int threadOutRow0 = outBlockRow + ty * OUTPT;
    const int threadOutCol0 = outBlockCol + tx * OUTPT;

    // Shared tile origin in input coordinates (includes halo)
    const int sOriginRow = outBlockRow - FILTER_RAD;
    const int sOriginCol = outBlockCol - FILTER_RAD;

    // Cooperative fill of shared memory (linearized index, stride by nThreads)
    const int sSize = S_WIDTH * S_HEIGHT;
    for (int idx = tid; idx < sSize; idx += nThreads) {
        const int r = idx / S_WIDTH;
        const int c = idx % S_WIDTH;
        const int inRow = sOriginRow + r;
        const int inCol = sOriginCol + c;
        float v = 0.0f;
        if ((unsigned)inRow < (unsigned)N.height && (unsigned)inCol < (unsigned)N.width) {
            v = N.elements[inRow * N.width + inCol];
        }
        N_Sh[r][c] = v;
    }
    __syncthreads();

    // Local starting index inside shared memory for this thread's first output
    const int sx0 = tx * OUTPT + FILTER_RAD;
    const int sy0 = ty * OUTPT + FILTER_RAD;



    // compute OUTPT x OUTPT outputs per thread
    for (int oy = 0; oy < OUTPT; ++oy) {
        const int outRow = threadOutRow0 + oy;
        // optional early-out per-row if entire row outside image
        if ((unsigned)outRow >= (unsigned)P.height) continue;
        for (int ox = 0; ox < OUTPT; ++ox) {
            const int outCol = threadOutCol0 + ox;
            if ((unsigned)outCol >= (unsigned)P.width) continue; // single predicate per output
            float acc = 0.0f;
            const int py = sy0 + oy;
            const int px = sx0 + ox;
            #pragma unroll
            for (int ky = -FILTER_RAD; ky <= FILTER_RAD; ++ky) {
                #pragma unroll
                for (int kx = -FILTER_RAD; kx <= FILTER_RAD; ++kx) {
                    const float pix = N_Sh[py + ky][px + kx];
                    acc += pix * M_c[ky + FILTER_RAD][kx + FILTER_RAD];
                }
            }
            P.elements[outRow * P.width + outCol] = acc;
        }
    }
}

// __global__
// void conv_shared_to_regs(Matrix N, Matrix P) {
//     // shared tile: [row][col]
//     __shared__ float sTile[S_HEIGHT][S_WIDTH];

//     // block origin in OUTPUT space (top-left output produced by this block)
//     const int outBlockRow = blockIdx.y * TILE_OUT_DIM; // Y
//     const int outBlockCol = blockIdx.x * TILE_OUT_DIM; // X

//     // thread coords in block
//     const int tx = threadIdx.x; // 0..BLOCK_SIZE-1 (col in block)
//     const int ty = threadIdx.y; // 0..BLOCK_SIZE-1 (row in block)
//     const int threadsX = blockDim.x;
//     const int threadsY = blockDim.y;
//     const int tid = ty * threadsX + tx;
//     const int nThreads = threadsX * threadsY;

//     // shared tile origin in INPUT coords (top-left pixel loaded into sTile[0][0])
//     const int sOriginRow = outBlockRow - FILTER_RAD; // Y
//     const int sOriginCol = outBlockCol - FILTER_RAD; // X

//     // cooperative fill of sTile (row-major linear index)
//     const int sSize = S_WIDTH * S_HEIGHT;
//     for (int lin = tid; lin < sSize; lin += nThreads) {
//         const int r = lin / S_WIDTH; // row index inside sTile (Y)
//         const int c = lin % S_WIDTH; // col index inside sTile (X)
//         const int inRow = sOriginRow + r; // input Y
//         const int inCol = sOriginCol + c; // input X
//         float v = 0.0f;
//         if ((unsigned)inRow < (unsigned)N.height && (unsigned)inCol < (unsigned)N.width) {
//             v = N.elements[inRow * N.width + inCol]; // row-major load
//         }
//         sTile[r][c] = v;
//     }
//     __syncthreads();

//     // this thread's first output coordinates (global)
//     const int threadOutRow0 = outBlockRow + ty * OUTPT; // Y
//     const int threadOutCol0 = outBlockCol + tx * OUTPT; // X

//     // per-thread register patch of size P_DIM x P_DIM
//     float patch[P_DIM][P_DIM];

//     // top-left of patch inside sTile (row and col offsets)
//     const int sy0 = ty * OUTPT + FILTER_RAD; // row offset
//     const int sx0 = tx * OUTPT + FILTER_RAD; // col offset

//     // copy patch from shared memory into registers
//     #pragma unroll
//     for (int pr = 0; pr < P_DIM; ++pr) {
//         #pragma unroll
//         for (int pc = 0; pc < P_DIM; ++pc) {
//             patch[pr][pc] = sTile[sy0 + pr][sx0 + pc];
//         }
//     }

//     // compute OUTPT x OUTPT outputs using the register patch
//     for (int oy = 0; oy < OUTPT; ++oy) {
//         const int outRow = threadOutRow0 + oy; // Y
//         if ((unsigned)outRow >= (unsigned)P.height) continue; // single row predicate
//         for (int ox = 0; ox < OUTPT; ++ox) {
//             const int outCol = threadOutCol0 + ox; // X
//             if ((unsigned)outCol >= (unsigned)P.width) continue; // single col predicate
//             float acc = 0.0f;
//             const int pr0 = oy; // patch row start
//             const int pc0 = ox; // patch col start
//             #pragma unroll
//             for (int ky = 0; ky < FILTER_SIZE; ++ky) {
//                 #pragma unroll
//                 for (int kx = 0; kx < FILTER_SIZE; ++kx) {
//                     acc += patch[pr0 + ky][pc0 + kx] * M_c[ky][kx];
//                 }
//             }
//             P.elements[outRow * P.width + outCol] = acc;
//         }
//     }
// }

// __global__
// void conv_shared_to_regs_safe(Matrix N, Matrix P) {
//     __shared__ float sTile[S_HEIGHT][S_WIDTH];

//     const int outBlockRow = blockIdx.y * TILE_OUT_DIM; // Y
//     const int outBlockCol = blockIdx.x * TILE_OUT_DIM; // X

//     const int tx = threadIdx.x;
//     const int ty = threadIdx.y;
//     const int threadsX = blockDim.x;
//     const int threadsY = blockDim.y;
//     const int tid = ty * threadsX + tx;
//     const int nThreads = threadsX * threadsY;

//     const int sOriginRow = outBlockRow - FILTER_RAD; // Y
//     const int sOriginCol = outBlockCol - FILTER_RAD; // X

//     // Cooperative fill; elements outside image become 0.0f
//     const int sSize = S_WIDTH * S_HEIGHT;
//     for (int idx = tid; idx < sSize; idx += nThreads) {
//         const int r = idx / S_WIDTH; // row in sTile (Y)
//         const int c = idx % S_WIDTH; // col in sTile (X)
//         const int inRow = sOriginRow + r;
//         const int inCol = sOriginCol + c;
//         float v = 0.0f;
//         if ((unsigned)inRow < (unsigned)N.height && (unsigned)inCol < (unsigned)N.width) {
//             v = N.elements[inRow * N.width + inCol];
//         }
//         sTile[r][c] = v;
//     }
//     __syncthreads();

//     const int threadOutRow0 = outBlockRow + ty * OUTPT; // Y
//     const int threadOutCol0 = outBlockCol + tx * OUTPT; // X

//     // per-thread register patch
//     float patch[P_DIM][P_DIM];

//     // top-left inside sTile
//     const int sy0 = ty * OUTPT + FILTER_RAD; // row offset
//     const int sx0 = tx * OUTPT + FILTER_RAD; // col offset

//     // SAFE copy: check bounds against sTile extents before reading
//     #pragma unroll
//     for (int pr = 0; pr < P_DIM; ++pr) {
//         #pragma unroll
//         for (int pc = 0; pc < P_DIM; ++pc) {
//             const int sR = sy0 + pr;
//             const int sC = sx0 + pc;
//             // these tests protect against any host/compile-time mismatch or partial loads
//             if ((unsigned)sR < (unsigned)S_HEIGHT && (unsigned)sC < (unsigned)S_WIDTH) {
//                 patch[pr][pc] = sTile[sR][sC];
//             } else {
//                 patch[pr][pc] = 0.0f;
//             }
//         }
//     }

//     // compute outputs (single predicate per output)
//     for (int oy = 0; oy < OUTPT; ++oy) {
//         const int outRow = threadOutRow0 + oy;
//         if ((unsigned)outRow >= (unsigned)P.height) continue;
//         for (int ox = 0; ox < OUTPT; ++ox) {
//             const int outCol = threadOutCol0 + ox;
//             if ((unsigned)outCol >= (unsigned)P.width) continue;
//             float acc = 0.0f;
//             const int pr0 = oy;
//             const int pc0 = ox;
//             #pragma unroll
//             for (int ky = 0; ky < FILTER_SIZE; ++ky) {
//                 #pragma unroll
//                 for (int kx = 0; kx < FILTER_SIZE; ++kx) {
//                     acc += patch[pr0 + ky][pc0 + kx] * M_c[ky][kx];
//                 }
//             }
//             P.elements[outRow * P.width + outCol] = acc;
//         }
//     }
// }

__global__
void convolution_tiled_per_thread_vec(Matrix N, Matrix P) {
    // Shared tile (with halo), 16-byte aligned for vectorized stores
    __shared__ __align__(16) float N_Sh[S_HEIGHT][S_WIDTH];

    // Block origin in output coordinates (top-left output pixel for this block)
    const int outBlockRow = blockIdx.y * TILE_OUT_DIM;
    const int outBlockCol = blockIdx.x * TILE_OUT_DIM;

    // Thread info
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int threadsPerBlockX = blockDim.x;
    const int threadsPerBlockY = blockDim.y;
    const int tid = ty * threadsPerBlockX + tx;
    const int nThreads = threadsPerBlockX * threadsPerBlockY;

    // First output this thread will produce
    const int threadOutRow0 = outBlockRow + ty * OUTPT;
    const int threadOutCol0 = outBlockCol + tx * OUTPT;

    // Shared tile origin in input coords (includes halo)
    const int sOriginRow = outBlockRow - FILTER_RAD;
    const int sOriginCol = outBlockCol - FILTER_RAD;

    // Cooperative fill of shared memory
    const int sSize = S_WIDTH * S_HEIGHT;
    const bool canVecLoad = ((sOriginCol & 3) == 0) && ((S_WIDTH & 3) == 0);

    if (canVecLoad) {
        const int vecWidth = 4;
        const int sCols4 = S_WIDTH / vecWidth;
        const int sSize4 = S_HEIGHT * sCols4;

        for (int idx4 = tid; idx4 < sSize4; idx4 += nThreads) {
            const int r  = idx4 / sCols4;
            const int c4 = idx4 % sCols4;
            const int inRow = sOriginRow + r;
            const int inCol = sOriginCol + c4 * vecWidth;

            float4 v4 = make_float4(0.f, 0.f, 0.f, 0.f);
            if ((unsigned)inRow < (unsigned)N.height &&
                (unsigned)(inCol + 3) < (unsigned)N.width) {
                const float4* gptr = reinterpret_cast<const float4*>(&N.elements[inRow * N.width + inCol]);
                v4 = *gptr;
            } else {
                float tmp[4] = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int lane = 0; lane < 4; ++lane) {
                    const int ic = inCol + lane;
                    if ((unsigned)inRow < (unsigned)N.height &&
                        (unsigned)ic < (unsigned)N.width) {
                        tmp[lane] = N.elements[inRow * N.width + ic];
                    }
                }
                v4 = make_float4(tmp[0], tmp[1], tmp[2], tmp[3]);
            }

            float4* sptr = reinterpret_cast<float4*>(&N_Sh[r][c4 * vecWidth]);
            *sptr = v4;
        }
    } else {
        // Scalar fallback for misalignment cases
        for (int idx = tid; idx < sSize; idx += nThreads) {
            const int r = idx / S_WIDTH;
            const int c = idx % S_WIDTH;
            const int inRow = sOriginRow + r;
            const int inCol = sOriginCol + c;
            float v = 0.0f;
            if ((unsigned)inRow < (unsigned)N.height && (unsigned)inCol < (unsigned)N.width) {
                v = N.elements[inRow * N.width + inCol];
            }
            N_Sh[r][c] = v;
        }
    }

    __syncthreads();

    // Local starting index inside shared memory for this thread's first output
    const int sx0 = tx * OUTPT + FILTER_RAD;
    const int sy0 = ty * OUTPT + FILTER_RAD;

    // Compute OUTPT x OUTPT outputs per thread; vectorize stores along x (4-wide)
    for (int oy = 0; oy < OUTPT; ++oy) {
        const int outRow = threadOutRow0 + oy;
        if ((unsigned)outRow >= (unsigned)P.height) continue;

        for (int ox = 0; ox < OUTPT; ox += 4) {
            const int outCol = threadOutCol0 + ox;

            // Accumulate 4 adjacent outputs in registers
            float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
            const int py = sy0 + oy;
            const int px0 = sx0 + ox;

            #pragma unroll
            for (int ky = -FILTER_RAD; ky <= FILTER_RAD; ++ky) {
                #pragma unroll
                for (int kx = -FILTER_RAD; kx <= FILTER_RAD; ++kx) {
                    const float w  = M_c[ky + FILTER_RAD][kx + FILTER_RAD];
                    const int sry = py + ky;
                    const int scx = px0 + kx;

                    const float p0 = N_Sh[sry][scx + 0];
                    const float p1 = N_Sh[sry][scx + 1];
                    const float p2 = N_Sh[sry][scx + 2];
                    const float p3 = N_Sh[sry][scx + 3];

                    acc0 += p0 * w;
                    acc1 += p1 * w;
                    acc2 += p2 * w;
                    acc3 += p3 * w;
                }
            }

            // Vectorized store when aligned and within right border, otherwise scalar
            const bool canVecStore =
                ((P.width - outCol) >= 4) && ((outCol & 3) == 0);

            if (canVecStore) {
                float4 outv = make_float4(acc0, acc1, acc2, acc3);
                float4* gptr = reinterpret_cast<float4*>(&P.elements[outRow * P.width + outCol]);
                *gptr = outv;
            } else {
                if ((unsigned)outCol < (unsigned)P.width)
                    P.elements[outRow * P.width + outCol + 0] = acc0;
                if ((unsigned)(outCol + 1) < (unsigned)P.width)
                    P.elements[outRow * P.width + outCol + 1] = acc1;
                if ((unsigned)(outCol + 2) < (unsigned)P.width)
                    P.elements[outRow * P.width + outCol + 2] = acc2;
                if ((unsigned)(outCol + 3) < (unsigned)P.width)
                    P.elements[outRow * P.width + outCol + 3] = acc3;
            }
        }
    }
}