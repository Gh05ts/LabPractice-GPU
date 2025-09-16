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

void verify(float *A, float *B, int height, int width)
{
	const float relativeTolerance = 1e-6;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int index = (i *width + j) *3;
			float ref = 0.144 *A[index] + 0.587 *A[index + 1] + 0.299 *A[index + 2];
			float relativeError = (ref - B[i *width + j]) / ref;
			if (relativeError > relativeTolerance ||
				relativeError < -relativeTolerance)
			{
				printf("TEST FAILED\n\n");
				printf("\tRef: %f, GPU: %f\n", ref, B[i *width + j]);
				exit(0);
			}
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

	unsigned int image_height;
	unsigned int image_width;
	if (argc == 1)
	{
		image_height = 128;
		image_width = 128;
	}
	else if (argc == 3)
	{
		image_height = atoi(argv[1]);
		image_width = atoi(argv[2]);
	}
	else
	{
		printf("\n    Invalid input parameters!"
			"\n    Usage: ./image_to_gray           # Vector of size 10,000 is used"
			"\n    Usage: ./image_to_gray<h><w>   # Vector of size m is used"
			"\n");
		exit(0);
	}

	const size_t in_data_bytes = sizeof(float) *image_height *image_width * 3;
	const size_t out_data_bytes = sizeof(float) *image_height * image_width;
	// 3-channel image with H/W dimensions
	float *in_h = (float*) malloc(in_data_bytes);
	for (unsigned int i = 0; i < image_height; i++)
	{
		for (unsigned int j = 0; j < image_width; j++)
		{
			in_h[3 *(i *image_width + j)] = rand() % 255;
			in_h[3 *(i *image_width + j) + 1] = rand() % 255;
			in_h[3 *(i *image_width + j) + 2] = rand() % 255;
		}
	}

	// 1-channel image with H/W dimensions
	float *out_h = (float*) malloc(out_data_bytes);

	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));
	printf("    Image size = %u *%u\n", image_height, image_width);

	// Allocate device variables ----------------------------------------------
	printf("Allocating device variables...");
	fflush(stdout);
	startTime(&timer);
	//INSERT CODE HERE
	float *in_d, *out_d; 
	
	cudaMalloc((void **) &in_d, in_data_bytes);
	cudaMalloc((void **) &out_d, out_data_bytes);

	cudaDeviceSynchronize();
	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));

	// Copy host variables to device ------------------------------------------
	printf("Copying data from host to device...");
	fflush(stdout);
	startTime(&timer);
	//INSERT CODE HERE
	cudaMemcpy(in_d, in_h, in_data_bytes, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));

	// Launch kernel ----------------------------------------------------------
	printf("Launching kernel...");
	fflush(stdout);
	startTime(&timer);
	//INSERT CODE HERE
	dim3 dimGrid(ceil(image_height/16.0), ceil(image_width/16.0), 1);
	dim3 dimBlock(16, 16, 1);
	image2grayKernel<<<dimGrid, dimBlock>>>(in_d, out_d, image_height, image_width);

	cuda_ret = cudaDeviceSynchronize();
	if (cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));

	// Copy device variables from host ----------------------------------------
	printf("Copying data from device to host...");
	fflush(stdout);
	startTime(&timer);
	//INSERT CODE HERE
	cudaMemcpy(out_h, out_d, out_data_bytes, cudaMemcpyDeviceToHost);
	

	cudaDeviceSynchronize();
	stopTime(&timer);
	printf("%f s\n", elapsedTime(timer));

	// Verify correctness -----------------------------------------------------
	printf("Verifying results...");
	fflush(stdout);
	verify(in_h, out_h, image_height, image_width);

	// Free memory ------------------------------------------------------------
	free(in_h);
	free(out_h);
	//INSERT CODE HERE
	

	return 0;

}