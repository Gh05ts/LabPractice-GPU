/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

// Feel free to use other numbers for best performance
#define TILE_SIZE 16
#define BLOCK_SIZE 32

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float *C)
{

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;

    // __syncthreads();
    // printf("cRow -> %d", cRow);
    // __syncthreads();
    // printf("cCol -> %d", cCol);
    // __syncthreads();

    float sum = 0.0;
    if(row < m && col < n) {
        for(int i = 0; i < k; i++) {
            sum += A[row*k + i] * B[i*n + col];
        }
        C[row*n+col] = sum;
    }
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc, int testRound)
{
    if ((transa != 'N') && (transa != 'n'))
    {
        printf("unsupported value of 'transa'\n");
        return;
    }

    if ((transb != 'N') && (transb != 'n'))
    {
        printf("unsupported value of 'transb'\n");
        return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10))
    {
        printf("unsupported value of alpha\n");
        return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10))
    {
        printf("unsupported value of beta\n");
        return;
    }

    // Initialize thread block and kernel grid dimensions ----------------------
    // INSERT CODE HERE
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1)/ dimBlock.y);    

    for (int i = 0; i < testRound; i++) {
        // Invoke CUDA kernel --------------------------------------------------
        // INSERT CODE HERE
        mysgemm<<<dimGrid, dimBlock>>>(m, n, k, A, B, C);
        cudaDeviceSynchronize();
    }
}
