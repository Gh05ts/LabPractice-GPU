/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "tgemm_kernel.cu"
#include "support.h"

float* paddedMatrix(size_t matrixSize, int row, int col, int pCol) {
    float *matrix;
    cudaHostAlloc(&matrix, matrixSize * sizeof(float), cudaHostAllocDefault);
    memset(matrix, 0, sizeof(matrix));
    size_t j = 0;
    for(size_t i = 0; i < row; i++) {
        for(size_t k = 0; k < col; k++) {
            matrix[j + k] = (rand() % 100) / 100.00;
        } 
        j += pCol; 
    }
    return matrix;
}

int main(int argc,  char *argv[])
{
    const uint tile_size = 32;
    const uint thread_tile = 4;
    
    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem...");
    fflush(stdout);
    startTime(&timer);

    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    unsigned matArow, matAcol;
    unsigned matBrow, matBcol;
    unsigned testRound; // how many rounds to run
    
    int pMatARow, pMatACol, pMatBCol;
    size_t Ap_sz, Bp_sz, Cp_sz;

    if (argc == 1)
    {
        matArow = 1000;
        matAcol = matBrow = 1000;
        matBcol = 1000;
        testRound = 100;
    }
    else if (argc == 3)
    {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[1]);
        matBcol = atoi(argv[1]);
        testRound = atoi(argv[2]);
    }
    else if (argc == 5)
    {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[2]);
        matBcol = atoi(argv[3]);
        testRound = atoi(argv[4]);
    }
    else
    {
        printf("\n    Invalid input parameters!"
               "\n    Usage: ./sgemm-tiled                   # All matrices are 1000 x 1000"
               "\n    Usage: ./sgemm-tiled <m> <r>           # All matrices are m x m"
               "\n    Usage: ./sgemm-tiled <m> <k> <n> <r>   # A: m x k, B: k x n, C: m x n"
               "\n");
        exit(0);
    }

    pMatARow = ((matArow + (tile_size * thread_tile) - 1)/ (tile_size * thread_tile)) * (tile_size * thread_tile);
    pMatACol = ((matAcol + (tile_size * thread_tile) - 1)/ (tile_size * thread_tile)) * (tile_size * thread_tile);
    pMatBCol = ((matBcol + (tile_size * thread_tile) - 1)/ (tile_size * thread_tile)) * (tile_size * thread_tile);

    Ap_sz = pMatARow * pMatACol;
    Bp_sz = pMatACol * pMatBCol;
    Cp_sz = pMatARow * pMatBCol;

    A_h = paddedMatrix(Ap_sz, matArow, matAcol, pMatACol);
    B_h = paddedMatrix(Bp_sz, matBrow, matBcol, pMatBCol);

    cudaHostAlloc(&C_h, Cp_sz * sizeof(float), cudaHostAllocDefault);

    stopTime(&timer);
    printf("%f s\n", elapsedTime(timer));
    printf("    A: %u x %u\n    B: %u x %u\n    C: %u x %u\n", matArow, matAcol,
           matBrow, matBcol, matArow, matBcol);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables...");
    fflush(stdout);
    startTime(&timer);

    // INSERT CODE HERE
    cudaMalloc((void **) &A_d, Ap_sz * sizeof(float));
    cudaMalloc((void **) &B_d, Bp_sz * sizeof(float));
    cudaMalloc((void **) &C_d, Cp_sz * sizeof(float));

    cudaDeviceSynchronize();
    stopTime(&timer);
    printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device...");
    fflush(stdout);
    startTime(&timer);

    // INSERT CODE HERE
    cudaMemcpy(A_d, A_h, Ap_sz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, Bp_sz * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printf("%f s\n", elapsedTime(timer));

    // Launch kernel using standard sgemm interface ---------------------------
    printf("Launching kernel...");
    fflush(stdout);
    startTime(&timer);
    printf("\n");
    basicSgemm<tile_size, thread_tile>('N', 'N', pMatARow, pMatBCol, pMatACol, 1.0f, A_d, pMatARow, B_d, pMatACol, 0.0f, C_d, pMatACol, testRound);
    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess)
        // printf(cudaGetErrorString(cuda_ret));
        FATAL("Unable to launch kernel");
    stopTime(&timer);
    printf("%f s for %d rounds, i.e., %f/round\n", elapsedTime(timer), testRound, elapsedTime(timer) / testRound);

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host...");
    fflush(stdout);
    startTime(&timer);

    // INSERT CODE HERE
    cudaMemcpy(C_h, C_d, Cp_sz * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results...");
    fflush(stdout);

    verify(A_h, B_h, C_h, pMatARow, pMatACol, pMatBCol);

    // Free memory ------------------------------------------------------------

    cudaFreeHost(A_h);
    cudaFreeHost(B_h);
    cudaFreeHost(C_h);

    // INSERT CODE HERE
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}
