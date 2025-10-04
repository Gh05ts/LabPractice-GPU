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

#define TILE_SIZE 32
#define STREAMS 4

void printMatrix(float *matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        printf("%f ", matrix[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[])
{

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem...");
    fflush(stdout);
    startTime(&timer);

    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    size_t A_sz, B_sz, C_sz;
    unsigned matArow, matAcol;
    unsigned matBrow, matBcol;
    unsigned testRound; // how many rounds to run
    dim3 dim_grid, dim_block;
    
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

    pMatARow = ((matArow + TILE_SIZE - 1)/TILE_SIZE) * TILE_SIZE;
    pMatACol = ((matAcol + TILE_SIZE - 1)/TILE_SIZE) * TILE_SIZE;
    pMatBCol = ((matBcol + TILE_SIZE - 1)/TILE_SIZE) * TILE_SIZE;

    A_sz = matArow * matAcol;
    B_sz = matBrow * matBcol;
    C_sz = matArow * matBcol;

    Ap_sz = pMatARow * pMatACol;
    Bp_sz = pMatACol * pMatBCol;
    Cp_sz = pMatARow * pMatBCol;

    A_h = (float *)malloc(sizeof(float) * Ap_sz);
    for (unsigned int i = 0; i < A_sz; i++)
    {
        A_h[i] = (rand() % 100) / 100.00;
    }
    if(Ap_sz >  A_sz) {
        for (unsigned int i = A_sz; i < Ap_sz; i++)
        {
            A_h[i] = 0;
        }        
    }

    B_h = (float *)malloc(sizeof(float) * Bp_sz);
    for (unsigned int i = 0; i < B_sz; i++)
    {
        B_h[i] = (rand() % 100) / 100.00;
    }
    if(Bp_sz >  B_sz) {
        for (unsigned int i = B_sz; i < Bp_sz; i++)
        {
            B_h[i] = 0;        
        }
    }

    C_h = (float *)malloc(sizeof(float) * Cp_sz);

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
    // printMatrix(A_h, A_sz);
    // printMatrix(B_h, B_sz);
    printf("\n");
    basicSgemm('N', 'N', pMatARow, pMatBCol, pMatACol, 1.0f,
               A_d, pMatARow, B_d, pMatACol, 0.0f, C_d, pMatACol, testRound);

    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess)
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

    verify(A_h, B_h, C_h, matArow, matAcol, matBcol);

    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    // INSERT CODE HERE
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}
