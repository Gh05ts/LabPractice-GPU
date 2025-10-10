/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

using Vec4 = float4;

template<const uint TILE_SIZE, const uint THREAD_TILE>
__global__ 
void mysgemm(const int m, const int n, const int k, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {

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
  int blockRowStart = blockIdx.y * TILE_SIZE;
  int blockColStart = blockIdx.x * TILE_SIZE;

  int threadRowOff = threadIdx.y * THREAD_TILE;
  int threadColOff = threadIdx.x * THREAD_TILE;

  __shared__ float sA[TILE_SIZE][TILE_SIZE];
  __shared__ float sB[TILE_SIZE][TILE_SIZE];

  float Creg[THREAD_TILE][THREAD_TILE] = {{0}};

  for (int kTileStart = 0; kTileStart < k; kTileStart += TILE_SIZE) {
    #pragma unroll
    for(int i = 0; i < THREAD_TILE; ++i) {
      int sRow = threadRowOff + i;

      for(int v = 0; v < THREAD_TILE; v += 4) {
        int sCol = threadColOff + v;

        int aRow = blockRowStart + sRow;
        int aCol = kTileStart + sCol;
        Vec4 aVec = reinterpret_cast<const Vec4*>(A + aRow * k + aCol)[0];
        *reinterpret_cast<Vec4*>(&sA[sRow][sCol]) = aVec;

        int bRow = kTileStart + sRow;
        int bCol = blockColStart + sCol;
        Vec4 bVec = reinterpret_cast<const Vec4*>(B + bRow * n + bCol)[0];
        *reinterpret_cast<Vec4*>(&sB[sRow][sCol]) = bVec;
      }
    }
    __syncthreads();

    for (int kk = 0; kk < TILE_SIZE; ++kk) {
      float aReg[THREAD_TILE];
      float bReg[THREAD_TILE];

      for (int i = 0; i < THREAD_TILE; ++i) {
        aReg[i] = sA[threadRowOff + i][kk];
      }

      for (int j = 0; j < THREAD_TILE; ++j) {
        bReg[j] = sB[kk][threadColOff + j];
      }

      for (int i = 0; i < THREAD_TILE; ++i) {
        for (int j = 0; j < THREAD_TILE; ++j) {
          Creg[i][j] += aReg[i] * bReg[j];
        }
      }
    }
    __syncthreads();
  }

  for (int i = 0; i < THREAD_TILE; ++i) {
    int row = blockRowStart + threadRowOff + i;

    float* Crow  = C + row * n + blockColStart;
    float4* Crow4 = reinterpret_cast<float4*>(Crow);

    for (int v4 = 0; v4 < THREAD_TILE/4; ++v4) {
      int idx4 = threadIdx.x * (THREAD_TILE/4) + v4;
      Crow4[idx4] = make_float4(Creg[i][v4*4 + 0], Creg[i][v4*4 + 1], Creg[i][v4*4 + 2], Creg[i][v4*4 + 3]);
    }
  }  
}

template<const uint TILE_SIZE, const uint THREAD_TILE>
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
  dim3 dimBlock(TILE_SIZE / THREAD_TILE, TILE_SIZE / THREAD_TILE);
  dim3 dimGrid((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1)/ TILE_SIZE);    

  for (int i = 0; i < testRound; i++) {
    // Invoke CUDA kernel --------------------------------------------------
    // INSERT CODE HERE
    mysgemm<TILE_SIZE, THREAD_TILE><<<dimGrid, dimBlock>>>(m, n, k, A, B, C);
    cudaDeviceSynchronize();
  }
}

