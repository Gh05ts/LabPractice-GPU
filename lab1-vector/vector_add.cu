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

	float *A_h = (float*) malloc(sizeof(float) *n);
	for (unsigned int i = 0; i < n; i++)
	{
		A_h[i] = (rand() % 100) / 100.00;
	}

	float *B_h = (float*) malloc(sizeof(float) *n);
	for (unsigned int i = 0; i < n; i++)
	{
		B_h[i] = (rand() % 100) / 100.00;
	}

	float *C_h = (float*) malloc(sizeof(float) *n);

	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));
	printf("    Vector size = %u\n", n);

	// Allocate device variables ----------------------------------------------
	printf("Allocating device variables...");
	fflush(stdout);
	startTime(&timer);
	//INSERT CODE HERE
	int size = n * sizeof(float);
	float *A_d, *B_d, *C_d;

	cudaMalloc((void **) &A_d, size);
	cudaMalloc((void **) &B_d, size);
	cudaMalloc((void **) &C_d, size);
	

	cudaDeviceSynchronize();
	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));

	// Copy host variables to device ------------------------------------------
	printf("Copying data from host to device...");
	fflush(stdout);
	startTime(&timer);
	//INSERT CODE HERE
	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));

	// Launch kernel ----------------------------------------------------------
	printf("Launching kernel...");
	fflush(stdout);
	startTime(&timer);
	//INSERT CODE HERE
	vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

	cuda_ret = cudaDeviceSynchronize();
	if (cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));

	    // Compute floating point operations per second.
    int nFlops = n;
    float nFlopsPerSec = nFlops/elapsedTime(timer);
    float nGFlopsPerSec = nFlopsPerSec*1e-9;

	// Compute transfer rates.
    int nBytes = 3*4*n; // 2N words in, 1N word out
    double nBytesPerSec = nBytes/elapsedTime(timer);
    double nGBytesPerSec = nBytesPerSec*1e-9;

    printf( "Time: %lf (sec), GFlopsS: %lf, GBytesS: %lf\n", 
             elapsedTime(timer), nGFlopsPerSec, nGBytesPerSec);

	// Copy device variables from host ----------------------------------------
	printf("Copying data from device to host...");
	fflush(stdout);
	startTime(&timer);
	//INSERT CODE HERE
	cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));

	// Verify correctness -----------------------------------------------------
	printf("Verifying results...");
	fflush(stdout);
	verify(A_h, B_h, C_h, n);


	// Free memory ------------------------------------------------------------
	free(A_h);
	free(B_h);
	free(C_h);
	//INSERT CODE HERE
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);

	return 0;

}