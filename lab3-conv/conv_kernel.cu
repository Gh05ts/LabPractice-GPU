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

    constexpr const int filter_rad = (FILTER_SIZE - 1) / 2;
    __shared__ float N_Sh[BLOCK_SIZE][BLOCK_SIZE];

    int row = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    int col = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    int srow = row - filter_rad;
    int scol = col - filter_rad;

    for(int i = threadIdx.y; i < TILE_SIZE; i += BLOCK_SIZE) {
        for(int j = threadIdx.x; j < TILE_SIZE; j += BLOCK_SIZE) {
            int ssrow = srow + i;
            int sscol = scol + j;

            if(ssrow >= 0 && ssrow < N.height && sscol >= 0 && sscol < N.width) {
                N_Sh[ssrow][sscol] = N.elements[ssrow * N.width + sscol];
            } else {
                N_Sh[ssrow][sscol] = 0.0f;
            }
        }
    }
    __syncthreads();

    if(row < N.height && col < N.width) {
        float acc = 0.0f;
        
    }
}