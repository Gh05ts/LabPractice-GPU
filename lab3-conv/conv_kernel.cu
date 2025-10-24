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
            float pixel = N.elements[inpX + inpY * N.width];
            float weight = M_c[filY][filX];
            Pvalue += pixel * weight;
        }
    }    
    P.elements[outRow * P.width + outCol] = Pvalue;
}