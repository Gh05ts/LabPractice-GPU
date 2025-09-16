/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
#include <stdio.h>
#include "support.h"
#include "kernel.cu"

// #include "npp.h" 
// #include "nppi.h"

void verify(float *A, float *B, float *C, int n)
{
	const float relativeTolerance = 1e-6;
	for (int i = 0; i < n; i++)
	{
		float sum = A[i] + B[i];
		float relativeError = (sum - C[i]) / sum;
		if (relativeError > relativeTolerance ||
			relativeError < -relativeTolerance)
		{
			printf("TEST FAILED\n\n");
			exit(0);
		}
	}

	printf("TEST PASSED\n\n");
}

int main(int argc, char **argv)
{
	Timer timer;
	cudaError_t cuda_ret;

	// Initialize host variables ----------------------------------------------

	printf("\nSetting up the problem...");
	fflush(stdout);
	startTime(&timer);

	unsigned int n;
	if (argc == 1)
	{
		n = 10000;
	}
	else if (argc == 2)
	{
		n = atoi(argv[1]);
	}
	else
	{
		printf("\n    Invalid input parameters!"
			"\n    Usage: ./vector_add       # Vector of size 10,000 is used"
			"\n    Usage: ./vector_add <m>   # Vector of size m is used"
			"\n");
		exit(0);
	}

	// float *A_h = (float*) malloc(sizeof(float) *n);
	// for (unsigned int i = 0; i < n; i++)
	// {
	// 	A_h[i] = (rand() % 100) / 100.00;
	// }

	// float *B_h = (float*) malloc(sizeof(float) *n);
	// for (unsigned int i = 0; i < n; i++)
	// {
	// 	B_h[i] = (rand() % 100) / 100.00;
	// }

	// float *C_h = (float*) malloc(sizeof(float) *n);

	float *A_h; 
	cudaHostAlloc(&A_h, sizeof(float) * n, cudaHostAllocDefault);
	for (unsigned int i = 0; i < n; i++)
	{
		A_h[i] = (rand() % 100) / 100.00;
	}


	float *B_h;
	cudaHostAlloc(&B_h, sizeof(float) * n, cudaHostAllocDefault);
	for (unsigned int i = 0; i < n; i++)
	{
		B_h[i] = (rand() % 100) / 100.00;
	}

	float *C_h;
	cudaHostAlloc(&C_h, sizeof(float) * n, cudaHostAllocDefault);
	for (unsigned int i = 0; i < n; i++)
	{
		C_h[i] = (rand() % 100) / 100.00;
	}

	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));
	printf("    Vector size = %u\n", n);

	// Allocate device variables ----------------------------------------------
	printf("Allocating device variables...");
	fflush(stdout);
	startTime(&timer);
	//INSERT CODE HERE
	const int V4_N     = n / 4;
	int size = V4_N * sizeof(float4);
	const int BLOCK_SZ = 256;
	const int GRID_SZ = (V4_N + BLOCK_SZ - 1) / BLOCK_SZ;

	int minGrid, optBlock;
	cudaOccupancyMaxPotentialBlockSize(
		&minGrid,      // returns the minimum grid size needed to achieve max occupancy
		&optBlock,     // returns the block size that achieves max occupancy
		vecAdd4,       // your kernel pointer
		0,             // dynamic shared mem per block
		V4_N           // maximum threads you’ll launch
	);
	// Then:
	int gridSize = (V4_N + optBlock - 1) / optBlock;

	printf("optBlock: %d\n", optBlock);
	printf("optGrid: %d\n", gridSize);
	fflush(stdout);

	// float *A_d, *B_d, *C_d;
	float4 *d_A4, *d_B4, *d_C4;

	cudaMalloc((void **) &d_A4, size);
	cudaMalloc((void **) &d_B4, size);
	cudaMalloc((void **) &d_C4, size);
	
	cudaStream_t stream;
    cudaStreamCreate(&stream);

	// cudaDeviceSynchronize();
	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));

	// Copy host variables to device ------------------------------------------
	printf("Copying data from host to device...");
	fflush(stdout);
	startTime(&timer);
	//INSERT CODE HERE
	// cudaMemcpy(d_A4, A_h, size, cudaMemcpyHostToDevice);
	// cudaMemcpy(d_B4, B_h, size, cudaMemcpyHostToDevice);

	cudaMemcpyAsync(d_A4, A_h, n * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_B4, B_h, n * sizeof(float), cudaMemcpyHostToDevice, stream);

	// nppiRGBToGray_16s_AC4C1R();

	// cudaDeviceSynchronize();
	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));

	// Launch kernel ----------------------------------------------------------
	printf("Launching kernel...");
	fflush(stdout);
	startTime(&timer);
	//INSERT CODE HERE
	// vecAddKernel<<<ceil(n/256.0), 256>>>(d_A4, d_B4, d_C4, n);
	vecAdd4<<<gridSize, optBlock>>>(d_A4, d_B4, d_C4, V4_N);

	// cuda_ret = cudaDeviceSynchronize();
	// if (cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));

	//     // Compute floating point operations per second.
    // int nFlops = n;
    // float nFlopsPerSec = nFlops/elapsedTime(timer);
    // float nGFlopsPerSec = nFlopsPerSec*1e-9;

	// // Compute transfer rates.
    // int nBytes = 3*4*n; // 2N words in, 1N word out
    // double nBytesPerSec = nBytes/elapsedTime(timer);
    // double nGBytesPerSec = nBytesPerSec*1e-9;

    // printf( "Time: %lf (sec), GFlopsS: %lf, GBytesS: %lf\n", 
    //          elapsedTime(timer), nGFlopsPerSec, nGBytesPerSec);

	// Copy device variables from host ----------------------------------------
	printf("Copying data from device to host...");
	fflush(stdout);
	startTime(&timer);
	//INSERT CODE HERE
	// cudaMemcpy(C_h, d_C4, size, cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(C_h, d_C4, n * sizeof(float), cudaMemcpyDeviceToHost, stream);

	cudaStreamSynchronize(stream);

	// cudaDeviceSynchronize();
	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));

	// Verify correctness -----------------------------------------------------
	printf("Verifying results...");
	fflush(stdout);
	verify(A_h, B_h, C_h, n);


	// Free memory ------------------------------------------------------------
	cudaStreamDestroy(stream);
	
	// free(A_h);
	// free(B_h);
	// free(C_h);
	cudaFreeHost(A_h);
	cudaFreeHost(B_h);
	cudaFreeHost(C_h);
	//INSERT CODE HERE
	cudaFree(d_A4);
	cudaFree(d_B4);
	cudaFree(d_C4);

	return 0;

}