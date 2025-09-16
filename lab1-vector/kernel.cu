/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

__global__ 
void vecAddKernel(float *A, float *B, float *C, int n)
{
	// Calculate global thread index based on the block and thread indices ----
	//INSERT KERNEL CODE HERE
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	// Use global index to determine which elements to read, add, and write ---
	//INSERT KERNEL CODE HERE, BE CAREFUL FOR CORNER CASE!!!
	if (i < n) {
		C[i] = A[i] + B[i];
	}
}

__global__ 
void image2grayKernel(float *in, float *out, int height, int width)
{
	// Calculate global thread index based on the block and thread indices ----
	//INSERT KERNEL CODE HERE
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	if(col < width && row < height) {
		int grayOffset = row*width + col;

		int rgbOffset = grayOffset*3; // channels = 3
		unsigned char r = in[rgbOffset];
		unsigned char g = in[rgbOffset + 1];
		unsigned char b = in[rgbOffset + 2];

		out[grayOffset] = 0.144f*r + 0.587f*g + 0.299f*b;
	}
	// Use global index to determine which elements to read, add, and write ---
	//INSERT KERNEL CODE HERE, BE CAREFUL FOR CORNER CASE!!!
	
}