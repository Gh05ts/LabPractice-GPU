/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include "support.h"
#include "stdio.h"

__constant__ float M_c[FILTER_SIZE][FILTER_SIZE] = {
    {1, 1, 1},
    // , 1, 1}, 
    {1, 1, 1},
    // , 1, 1}, 
    {1, 1, 1}
    // , 1, 1}
    // , 
    // {1, 1, 1, 1, 1}, 
    // {1, 1, 1, 1, 1}
};

__device__
void printMatrixDev(Matrix M) {
    for(int i = 0; i < M.height; i++) {
        int x = i * M.width;
        for(int j = 0; j < M.width; j++) {
            printf("%f, ", M.elements[x + j]);
        }
        printf("\n");
    }
}

__global__ 
void convolution(cudaTextureObject_t N, Matrix P) {
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

    if (outCol >= P.width || outRow >= P.height) return;

    float Pvalue = 0.0f;
    for (int fy = -filter_rad; fy <= filter_rad; fy++) {
        for (int fx = -filter_rad; fx <= filter_rad; fx++) {
            float pixel;
            if (outRow + fy < 0 || outRow + fy >= P.height || outCol + fx < 0 || outCol + fx >= P.width) {
                pixel = 0.0f; // explicit zero padding
            } else {
                pixel = tex2D<float>(N, outCol + fx, fy + outRow); // safe to sample
            }
            float weight = M_c[fy + filter_rad][fx + filter_rad];
            // printf("x: %d, y: %d\ntexX: %d, texY: %d, McY: %d, McX: %d, weight: %f, elem: %f\n", outCol, outRow, outCol + fx, outRow + fy, fy + filter_rad, fx + filter_rad, weight, pixel);
            // printf("", );
            // printf("", );
            // printf("", weight);
            // printf("", );
            Pvalue += pixel * weight;
        }
    }    
    P.elements[outRow * P.width + outCol] = Pvalue;
    // printMatrixDev(P);
}
