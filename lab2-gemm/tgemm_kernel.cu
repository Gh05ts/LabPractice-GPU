/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

// Feel free to use other numbers for best performance
#define TILE_SIZE 32
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
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    // 2
    // int lane = threadIdx.y * blockDim.x + threadIdx.x;
    // int local_y = lane / TILE_SIZE;
    // int local_x = lane % TILE_SIZE;

    // 3
    int local_x = threadIdx.x; // varies fastest across a warp
    int local_y = threadIdx.y;

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // 1
    // int row = blockRow * TILE_SIZE + threadIdx.y;
    // int col = blockCol * TILE_SIZE + threadIdx.x;

    // 2
    // int row = blockRow * TILE_SIZE + local_y;
    // int col = blockCol * TILE_SIZE + local_x;

    // const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    // const uint col = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("", TILE_SIZE);

    // __syncthreads();
    // printf("cRow -> %d", cRow);
    // __syncthreads();
    // printf("cCol -> %d", cCol);
    // __syncthreads();

    // 3
    int row = blockRow * TILE_SIZE + local_y;
    int col = blockCol * TILE_SIZE + local_x;

    float sum = 0.0;
    int tiles = k / TILE_SIZE;

    for(int t = 0; t < tiles; t++) {
        // int A_off = (blockRow * TILE_SIZE) * k + t * TILE_SIZE;
        // int B_off = (t * TILE_SIZE) * n + blockCol * TILE_SIZE;

        // sA[threadIdx.y][threadIdx.x] = A[A_off + threadIdx.y * k + threadIdx.x];
        // sB[threadIdx.y][threadIdx.x] = B[B_off + threadIdx.y * n + threadIdx.x];

        // 2
        // int a_col = t * TILE_SIZE + local_x;
        // int a_index = row * k + a_col;
        // sA[local_y][local_x] = A[a_index];

        // int b_row = t*TILE_SIZE + local_y;
        // int b_col = blockCol * TILE_SIZE + local_x;
        // int b_index = b_row * n + b_col;
        // sB[local_y][local_x] = B[b_index];
        
        int a_col = t * TILE_SIZE + local_x;
        int a_index = row * k + a_col;
        sA[local_y][local_x] = A[a_index];

        int b_row = t * TILE_SIZE + local_y;
        int b_col = blockCol * TILE_SIZE + local_x;
        int b_index = b_row * n + b_col;
        sB[local_y][local_x] = B[b_index];
        __syncthreads();

        #pragma unroll
        for(int i = 0; i < TILE_SIZE; i++) {
            sum += sA[local_y][i] * sB[i][local_x];
        }

        __syncthreads();
    }
    C[row*n+col] = sum;
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
