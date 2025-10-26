/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include "support.h"
#include "conv_kernel.cu"

bool checkTex(int imageHeight, int imageWidth) {
    return imageHeight < 128 && imageWidth < 128;
}

int main(int argc, char *argv[])
{
    Timer timer;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem...");
    fflush(stdout);
    startTime(&timer);

    Matrix M_h, N_h, P_h; // M: filter, N: input image, P: output image
    Matrix P_d, N_d; 
    cudaArray *cu;
    cudaTextureObject_t Nt_d;
    unsigned imageHeight, imageWidth;
    unsigned testRound; // how many rounds to run
    cudaError_t cuda_ret;
    dim3 dim_grid, dim_block;

    /* Read image dimensions */
    if (argc == 1)
    {
        imageHeight = 600;
        imageWidth = 1000;
        testRound = 100;
    }
    else if (argc == 3)
    {
        imageHeight = atoi(argv[1]);
        imageWidth = atoi(argv[1]);
        testRound = atoi(argv[2]);
    }
    else if (argc == 4)
    {
        imageHeight = atoi(argv[1]);
        imageWidth = atoi(argv[2]);
        testRound = atoi(argv[3]);
    }
    else
    {
        printf("\n    Invalid input parameters!"
               "\n    Usage: ./convolution             # Image is 600 x 1000"
               "\n    Usage: ./convolution <m> <r>     # Image is m x m"
               "\n    Usage: ./convolution <m> <n> <r> # Image is m x n"
               "\n");
        exit(0);
    }

    if(!checkTex(imageHeight, imageWidth)) {
        imageWidth += 128 - imageWidth % 128;
    }

    /* Allocate host memory */
    M_h = allocateMatrix(FILTER_SIZE, FILTER_SIZE);
    N_h = allocateMatrix(imageHeight, imageWidth);
    P_h = allocateMatrix(imageHeight, imageWidth);

    /* Initialize filter and images */
    initMatrix(M_h);
    initMatrix(N_h);

    stopTime(&timer);
    printf("%f s\n", elapsedTime(timer));
    printf("    Image: %u x %u\n", imageHeight, imageWidth);
    printf("    Mask: %u x %u\n", FILTER_SIZE, FILTER_SIZE);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables...");
    fflush(stdout);
    startTime(&timer);

    P_d = allocateDeviceMatrix(imageHeight, imageWidth);
    if(checkTex(imageHeight, imageWidth)) {
        cu = allocateDeviceArray(imageHeight, imageWidth);
    } else {
        N_d = allocateDeviceMatrix(imageHeight, imageWidth);
    }

    cudaDeviceSynchronize();
    stopTime(&timer);
    printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device...");
    fflush(stdout);
    startTime(&timer);

    /* Copy image to device global memory */
    if (checkTex(imageHeight, imageWidth)) {
        Nt_d = allocateTex(cu, N_h, imageHeight, imageWidth);
    } else {
        copyToDeviceMatrix(N_d, N_h);
    }

    /* Copy mask to device constant memory */
    // INSERT CODE HERE
    cuda_ret = cudaMemcpyToSymbol(M_c, M_h.elements, sizeof(float) * M_h.height * M_h.width, 0, cudaMemcpyHostToDevice);

    if (cuda_ret != cudaSuccess)
        FATAL("Unable to copy to constant memory");

    cudaDeviceSynchronize();
    stopTime(&timer);
    printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------
    printf("Launching kernel...");
    fflush(stdout);
    startTime(&timer);

    // INSERT CODE HERE
    dim_block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    int tileW = BLOCK_SIZE * OUTPT;
    int tileH = BLOCK_SIZE * OUTPT;
    dim3 grid((imageWidth + tileW - 1) / tileW, (imageHeight + tileH - 1) / tileH);
    dim_grid = dim3((imageWidth + (BLOCK_SIZE*1 - 1)) / BLOCK_SIZE*1, (imageHeight + (BLOCK_SIZE*1 - 1)) / BLOCK_SIZE*1);

    for (int i = 0; i < testRound; i++)
    {
        // INSERT CODE HERE
        // Call kernel function
        if(checkTex(imageHeight, imageWidth)) {
            convolutionTex<<<dim_grid, dim_block>>>(Nt_d, P_d);
        } else {
            convolution_tiled_per_thread_vec<<<grid, dim_block>>>(N_d, P_d);
        }
        cuda_ret = cudaDeviceSynchronize();
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error: %s\n", cudaGetErrorString(err));
        // Handle the error (e.g., exit, log, clean up resources)
    }
    if (cuda_ret != cudaSuccess)
        FATAL("Unable to launch/execute kernel");

    // cudaDeviceSynchronize();
    stopTime(&timer);
    printf("%f s for %d round, i.e., %f/round\n", elapsedTime(timer), testRound, elapsedTime(timer) / testRound);

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host...");
    fflush(stdout);
    startTime(&timer);

    copyFromDeviceMatrix(P_h, P_d);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results...");
    fflush(stdout);

    verify(M_h, N_h, P_h);

    // Free memory ------------------------------------------------------------

    freeMatrix(M_h);
    freeMatrix(N_h);
    freeMatrix(P_h);

    cudaDestroyTextureObject(Nt_d);
    cudaFreeArray(cu);

    freeDeviceMatrix(P_d);

    return 0;
}
