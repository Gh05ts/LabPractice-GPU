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

// #define OUTPT 2
constexpr int filter_rad = (FILTER_SIZE - 1) / 2;
// constexpr int PATCH = OUTPT + 2*filter_rad;

__global__ 
void convolution(cudaTextureObject_t N, Matrix P) {
    /********************************************************************
    Determine input and output indexes of each thread
    Load a tile of the input image to shared memory
    Apply the filter on the input image tile
    Write the compute values to the output image at the correct indexes
    ********************************************************************/

    // constexpr const int filter_rad = (FILTER_SIZE - 1) / 2;

    // int outBlockX = blockIdx.x * (blockDim.x * OUTPT);
    // int outBlockY = blockIdx.y * (blockDim.y * OUTPT);

    // thread-local output origin (top-left of the thread's 4x4 outputs)
    // int tx = threadIdx.x;
    // int ty = threadIdx.y;
    // int threadOutX0 = outBlockX + tx * OUTPT;
    // int threadOutY0 = outBlockY + ty * OUTPT;

    // top-left of the patch to load from input (includes halo)
    // int patchX0 = threadOutX0 - filter_rad;
    // int patchY0 = threadOutY0 - filter_rad;

    // INSERT KERNEL CODE HERE
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    if (outCol >= P.width || outRow >= P.height) return;

    float Pvalue = 0.0f;
    for (int fy = -filter_rad; fy <= filter_rad; fy++) {
        for (int fx = -filter_rad; fx <= filter_rad; fx++) {
            int inpX = outCol + fx;
            int inpY = fy + outRow;
            int filY = fy + filter_rad;
            int filX = fx + filter_rad;
            float pixel = tex2D<float>(N, inpX, inpY);
            float weight = M_c[filY][filX];
            Pvalue += pixel * weight;
        }
    }    
    P.elements[outRow * P.width + outCol] = Pvalue;
    // float patch[PATCH][PATCH];

    // Load the entire PATCH x PATCH into registers with bounds check
    // Each thread loads its own patch; texture reads are used for memory access
    // #pragma unroll
    // for (int r = 0; r < PATCH; ++r) {
    //     int iy = patchY0 + r;
    //     for (int c = 0; c < PATCH; ++c) {
    //         int ix = patchX0 + c;
    //         patch[r][c] = tex2D<float>(N, ix, iy);
    //     }
    // }

    // Compute the 4x4 outputs from the patch and write to P.elements
    // if(threadOutX0 * OUTPT < P.width && threadOutY0 * OUTPT < P.height) {
    //     #pragma unroll
    //     for (int oy = 0; oy < OUTPT; ++oy) {
    //         int outY = threadOutY0 + oy;
    //         for (int ox = 0; ox < OUTPT; ++ox) {
    //             int outX = threadOutX0 + ox;
    //             float sum = 0.0f;
    //             int prow = oy;
    //             int pcol = ox;
    //             #pragma unroll
    //             for (int fy = 0; fy < FILTER_SIZE; ++fy) {
    //                 for (int fx = 0; fx < FILTER_SIZE; ++fx) {
    //                     sum += patch[prow + fy][pcol + fx] * M_c[fy][fx];
    //                 }
    //             }
    //             P.elements[outY * P.width + outX] = sum;
    //         }
    //     }        
    // } 
    // else {
    //     for (int oy = 0; oy < OUTPT; ++oy) {
    //         int outY = threadOutY0 + oy;
    //         if (outY >= P.height) return;
    //         for (int ox = 0; ox < OUTPT; ++ox) {
    //             int outX = threadOutX0 + ox;
    //             if (outX >= P.width) continue;
    
    //             float sum = 0.0f;
    //             int prow = oy;
    //             int pcol = ox;
    //             for (int fy = 0; fy < FILTER_SIZE; ++fy) {
    //                 for (int fx = 0; fx < FILTER_SIZE; ++fx) {
    //                     sum += patch[prow + fy][pcol + fx] * M_c[fy][fx];
    //                 }
    //             }
    //             P.elements[outY * P.width + outX] = sum;
    //         }
    //     }
    // }
}
