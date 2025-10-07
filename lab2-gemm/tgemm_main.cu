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
#define THREAD_TILE 4
#define STREAMS 4

void printMatrix(float *matrix, int matrixSize) {
    for (int i = 0; i < matrixSize; i++) {
        printf("%f ", matrix[i]);
    }
    printf("\n");
}

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

    pMatARow = ((matArow + (TILE_SIZE * THREAD_TILE) - 1)/ (TILE_SIZE * THREAD_TILE)) * (TILE_SIZE * THREAD_TILE);
    pMatACol = ((matAcol + (TILE_SIZE * THREAD_TILE) - 1)/ (TILE_SIZE * THREAD_TILE)) * (TILE_SIZE * THREAD_TILE);
    pMatBCol = ((matBcol + (TILE_SIZE * THREAD_TILE) - 1)/ (TILE_SIZE * THREAD_TILE)) * (TILE_SIZE * THREAD_TILE);

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

    cudaStream_t s[STREAMS];
    for(int i = 0; i < STREAMS; ++i) {
        cudaStreamCreate(&s[i]);
    }

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


/*
#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

int main() {
    int M = 8192;            // rows of A and C
    int K = 8192;            // cols of A, rows of B
    int N = 8192;            // cols of B and C

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    // Tile along M (rows of A/C)
    int tileM = 1024; // tune this
    int numTiles = (M + tileM - 1) / tileM;

    // Allocate pinned host memory
    float *A_h = nullptr, *B_h = nullptr, *C_h = nullptr;
    cudaMallocHost(&A_h, bytesA);
    cudaMallocHost(&B_h, bytesB);
    cudaMallocHost(&C_h, bytesC);

    // Initialize A_h and B_h as needed for tests
    // for (size_t i=0;i<... ) A_h[i] = ...; B_h[...] = ...;

    // Allocate device memory
    float *A_d_tile = nullptr, *B_d = nullptr, *C_d = nullptr;
    size_t tileBytesA = (size_t)tileM * K * sizeof(float); // max tile allocation
    cudaMalloc(&A_d_tile, tileBytesA);
    cudaMalloc(&B_d, bytesB);
    cudaMalloc(&C_d, bytesC);

    // Copy full B once
    cudaMemcpy(B_d, B_h, bytesB, cudaMemcpyHostToDevice);

    // Streams and events
    const int numStreams = 2;
    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; ++i) cudaStreamCreate(&streams[i]);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Block and grid sizes
    dim3 block(16, 16); // tune block size
    // grid.x depends on N, grid.y depends on tileM_actual per-tile

    int warmups = 1;
    int rounds = 10;
    for (int r = -warmups; r < rounds; ++r) {
        bool isWarmup = (r < 0);
        if (!isWarmup) cudaEventRecord(start, 0);

        for (int t = 0; t < numTiles; ++t) {
            int tileRowOffset = t * tileM;
            int tileM_actual = std::min(tileM, M - tileRowOffset);
            size_t thisABytes = (size_t)tileM_actual * K * sizeof(float);
            size_t thisCBytes = (size_t)tileM_actual * N * sizeof(float);

            int sidx = t % numStreams;

            // Async copy A tile (host -> device) into A_d_tile on stream sidx
            cudaMemcpyAsync(A_d_tile,
                            A_h + (size_t)tileRowOffset * K,
                            thisABytes,
                            cudaMemcpyHostToDevice,
                            streams[sidx]);

            // Launch kernel on same stream to preserve ordering
            dim3 grid((N + block.x - 1) / block.x,
                      (tileM_actual + block.y - 1) / block.y);
            matmul_naive_tile<<<grid, block, 0, streams[sidx]>>>(
                A_d_tile, B_d, C_d, M, K, N, tileRowOffset, tileM_actual);

            // Async copy result tile back (device -> host)
            cudaMemcpyAsync(C_h + (size_t)tileRowOffset * N,
                            C_d + (size_t)tileRowOffset * N,
                            thisCBytes,
                            cudaMemcpyDeviceToHost,
                            streams[sidx]);
        }

        // synchronize streams for the round
        for (int i = 0; i < numStreams; ++i) cudaStreamSynchronize(streams[i]);

        if (!isWarmup) {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            printf("Round %d elapsed: %f ms\n", r, ms);
        }
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (int i=0;i<numStreams;++i) cudaStreamDestroy(streams[i]);
    cudaFree(A_d_tile);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaFreeHost(A_h);
    cudaFreeHost(B_h);
    cudaFreeHost(C_h);
    return 0;
}

*/