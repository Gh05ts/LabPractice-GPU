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
void image2grayKernel(const float *in, float *out, int height, int width)
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

__global__
__launch_bounds__(256)
void vecAdd4(const float4* __restrict__ A, const float4* __restrict__ B, float4* __restrict__ C, const int vectorizedN) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < vectorizedN) {
		float4 a = A[idx];
		float4 b = B[idx];
		C[idx] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, + a.w + b.w);
	}

}

__global__
__launch_bounds__(256)
void image2grayKernelOpt(const float3* __restrict__ in, float* __restrict__ out, const int height, const int width)
{
	// Calculate global thread index based on the block and thread indices ----
	//INSERT KERNEL CODE HERE
	int col = blockIdx.x*blockDim.x + threadIdx.x; // * 4
	// int row = blockIdx.y*blockDim.y 

	if(col < width * height) {
		// int grayOffset = row*width + col;
		// int colOffset = 
		// int rgbOffset = grayOffset*3; // channels = 3
		// unsigned char r = in[rgbOffset];
		// unsigned char g = in[rgbOffset + 1];
		// unsigned char b = in[rgbOffset + 2];

		// printf("%f\n", in[col].x);
		// printf("%f\n", in[col].y);
		// printf("%f\n", in[col].z);
		// out[grayOffset] = 0.144f*in[grayOffset].x+ 0.587f*in[grayOffset].y + 0.299f*in[grayOffset].z;
		// make_float4();
		out[col] = 0.144f*in[col].x+ 0.587f*in[col].y + 0.299f*in[col].z;
		// out[col + 1] = 0.144f*in[col + 1].x+ 0.587f*in[col + 1].y + 0.299f*in[col + 1].z;
		// out[col + 1] = 0.144f*in[col + 2].x+ 0.587f*in[col + 2].y + 0.299f*in[col + 2].z;
		// out[col + 1] = 0.144f*in[col + 3].x+ 0.587f*in[col + 3].y + 0.299f*in[col + 3].z;
		// printf("%f/n/n/n", out[col]);
	}
	// Use global index to determine which elements to read, add, and write ---
	//INSERT KERNEL CODE HERE, BE CAREFUL FOR CORNER CASE!!!
	
}
