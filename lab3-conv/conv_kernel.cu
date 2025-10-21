/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include "support.h"

__constant__ float M_c[FILTER_SIZE][FILTER_SIZE];

__global__ void convolution(cudaTextureObject_t N, Matrix P)
{
    /********************************************************************
    Determine input and output indexes of each thread
    Load a tile of the input image to shared memory
    Apply the filter on the input image tile
    Write the compute values to the output image at the correct indexes
    ********************************************************************/

    constexpr int filter_rad = (FILTER_SIZE - 1) / 2;

    // INSERT KERNEL CODE HERE
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    float Pvalue = 0.0f;
    // for(int fRow = 0; fRow < 2 * (r + 1)) {

    // }
    
}
