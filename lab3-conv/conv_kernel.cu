/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include "support.h"

__constant__ float M_c[FILTER_SIZE][FILTER_SIZE];

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

        #pragma unroll
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
    __shared__ float N_Sh[S_HEIGHT][S_WIDTH];

    const int outBlockRow = blockIdx.y * TILE_OUT_DIM;
    const int outBlockCol = blockIdx.x * TILE_OUT_DIM;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int threadsPerBlockX = blockDim.x;
    const int threadsPerBlockY = blockDim.y;
    const int tid = ty * threadsPerBlockX + tx;
    const int nThreads = threadsPerBlockX * threadsPerBlockY;

    const int threadOutRow0 = outBlockRow + ty * OUTPT;
    const int threadOutCol0 = outBlockCol + tx * OUTPT;

    const int sOriginRow = outBlockRow - FILTER_RAD;
    const int sOriginCol = outBlockCol - FILTER_RAD;

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

    const int sx0 = tx * OUTPT + FILTER_RAD;
    const int sy0 = ty * OUTPT + FILTER_RAD;

    for (int oy = 0; oy < OUTPT; ++oy) {
        const int outRow = threadOutRow0 + oy;
        if ((unsigned)outRow >= (unsigned)P.height) return;
        for (int ox = 0; ox < OUTPT; ++ox) {
            const int outCol = threadOutCol0 + ox;
            if ((unsigned)outCol >= (unsigned)P.width) continue;
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

__global__
void convolution_tiled_per_thread_vec(Matrix __restrict__ N, Matrix __restrict__ P) {
    __shared__ __align__(16) float N_Sh[S_HEIGHT][S_WIDTH];

    const int outBlockRow = blockIdx.y * TILE_OUT_DIM;
    const int outBlockCol = blockIdx.x * TILE_OUT_DIM;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int threadsPerBlockX = blockDim.x;
    const int threadsPerBlockY = blockDim.y;
    const int tid = ty * threadsPerBlockX + tx;
    const int nThreads = threadsPerBlockX * threadsPerBlockY;

    const int threadOutRow0 = outBlockRow + ty * OUTPT;
    const int threadOutCol0 = outBlockCol + tx * OUTPT;

    const int sOriginRow = outBlockRow - FILTER_RAD;
    const int sOriginCol = outBlockCol - FILTER_RAD;

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

    const int sx0 = tx * OUTPT + FILTER_RAD;
    const int sy0 = ty * OUTPT + FILTER_RAD;

    for (int oy = 0; oy < OUTPT; ++oy) {
        const int outRow = threadOutRow0 + oy;
        if ((unsigned)outRow >= (unsigned)P.height) continue;

        for (int ox = 0; ox < OUTPT; ox += 4) {
            const int outCol = threadOutCol0 + ox;

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

            float4 outv = make_float4(acc0, acc1, acc2, acc3);
            float4* gptr = reinterpret_cast<float4*>(&P.elements[outRow * P.width + outCol]);
            *gptr = outv;
        }
    }
}
