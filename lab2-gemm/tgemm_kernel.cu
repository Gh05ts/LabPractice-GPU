/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

// Feel free to use other numbers for best performance
#define TILE_SIZE 32
#define THREAD_TILE 4
#define BLOCK_SIZE 32

using Vec4 = float4;

__global__ 
void mysgemm(int m, int n, int k, const float *A, const float *B, float *C) {

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
      float aVals[THREAD_TILE];
      float bVals[THREAD_TILE];

      for (int i = 0; i < THREAD_TILE; ++i) {
        aVals[i] = sA[threadRowOff + i][kk];
      }

      for (int j = 0; j < THREAD_TILE; ++j) {
        bVals[j] = sB[kk][threadColOff + j];
      }

      for (int i = 0; i < THREAD_TILE; ++i) {
        for (int j = 0; j < THREAD_TILE; ++j) {
          Creg[i][j] += aVals[i] * bVals[j];
        }
      }
    }
    __syncthreads();
  }

  for (int i = 0; i < THREAD_TILE; ++i) {
    int r = blockRowStart + threadRowOff + i;

    float* Crow  = C + r * n + blockColStart;
    float4* Crow4 = reinterpret_cast<float4*>(Crow);

    for (int v4 = 0; v4 < THREAD_TILE/4; ++v4) {
      int idx4 = threadIdx.x * (THREAD_TILE/4) + v4;
      Crow4[idx4] = make_float4(Creg[i][v4*4 + 0], Creg[i][v4*4 + 1], Creg[i][v4*4 + 2], Creg[i][v4*4 + 3]);
    }
  }  
}

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
    mysgemm<<<dimGrid, dimBlock>>>(m, n, k, A, B, C);
    cudaDeviceSynchronize();
  }
}


/*
// gemm_variants.cu
// Compile: nvcc -O3 gemm_variants.cu -o gemm_variants

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cassert>

#define CHECK_CUDA(call) do { cudaError_t e = call; if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA err %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)

// Common config struct
struct GemmConfig {
  int BM, BN, BK;   // block tile sizes
  int TX, TY;       // threads per block (x,y)
  int R, S;         // per-thread micro-tile (rows, cols)
  bool useFloat4;   // vectorized global loads
  bool doubleBuffer; // use ping-pong shared buffers
};

// Two suggested configs
GemmConfig cfg_4090() {
  return GemmConfig{
    .BM = 128, .BN = 128, .BK = 32,
    .TX = 32, .TY = 8,               // 256 threads
    .R = 2, .S = 2,
    .useFloat4 = true,
    .doubleBuffer = true
  };
}
GemmConfig cfg_2090() {
  return GemmConfig{
    .BM = 64, .BN = 64, .BK = 16,
    .TX = 16, .TY = 16,              // 256 threads
    .R = 2, .S = 2,
    .useFloat4 = true,
    .doubleBuffer = false
  };
}

// Utility: ceil div
inline int cdiv(int a, int b){ return (a + b - 1) / b; }

// Variant kernels assume row-major A (M x K), row-major B (K x N), row-major C (M x N)
// lda = K, ldb = N, ldc = N

// Variant A: simple 2x2 micro-tile, scalar loads
template<int BM,int BN,int BK,int TX,int TY,int R,int S>
__global__ void gemm_variantA(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M,int N,int K,int lda,int ldb,int ldc)
{
  extern __shared__ float smem[]; // sA then sB concatenated: size BM*BK + BK*BN
  float* sA = smem;
  float* sB = smem + BM * BK;

  int block_m = blockIdx.y * BM;
  int block_n = blockIdx.x * BN;

  int tx = threadIdx.x; // 0..TX-1 (x = col-group)
  int ty = threadIdx.y; // 0..TY-1 (y = row-group)

  int row0 = block_m + ty * R;
  int col0 = block_n + tx * S;

  float acc[R][S];
  #pragma unroll
  for (int i=0;i<R;i++) for (int j=0;j<S;j++) acc[i][j]=0.0f;

  // cooperative load sizes
  const int rows_per_threadA = BM / TY; // assume divisible
  const int cols_per_threadB = BN / TX; // assume divisible
  const int elems_per_threadA = BK / TX; // assume divisible
  const int elems_per_threadB = BK / TY; // assume divisible

  for (int kb=0; kb<K; kb+=BK){
    // load A tile: BM x BK
    int start_rowA = block_m + ty * rows_per_threadA;
    int a_base_k = kb;
    #pragma unroll
    for (int r=0;r<rows_per_threadA;r++){
      int global_row = start_rowA + r;
      int sA_row = ty * rows_per_threadA + r;
      int sA_base = sA_row * BK;
      int a_glob_base = global_row*lda + a_base_k;
      #pragma unroll
      for (int e=0;e<elems_per_threadA;e++){
        int koff = tx * elems_per_threadA + e;
        float v = 0.0f;
        int gk = kb + koff;
        if (global_row < M && gk < K) v = A[a_glob_base + koff];
        sA[sA_base + koff] = v;
      }
    }

    // load B tile: BK x BN
    int start_colB = block_n + tx * cols_per_threadB;
    int b_base_k = kb;
    #pragma unroll
    for (int c=0;c<cols_per_threadB;c++){
      int global_col = start_colB + c;
      int sB_col = tx * cols_per_threadB + c;
      #pragma unroll
      for (int e=0;e<elems_per_threadB;e++){
        int koff = ty * elems_per_threadB + e;
        float v = 0.0f;
        int gk = kb + koff;
        if (gk < K && global_col < N) v = B[gk * ldb + global_col];
        sB[koff * BN + sB_col] = v;
      }
    }

    __syncthreads();

    #pragma unroll
    for (int k=0;k<BK;k++){
      float avals[R];
      #pragma unroll
      for (int i=0;i<R;i++){
        int srow = ty * R + i;
        avals[i] = sA[srow * BK + k];
      }
      float bvals[S];
      #pragma unroll
      for (int j=0;j<S;j++){
        int scol = tx * S + j;
        bvals[j] = sB[k * BN + scol];
      }
      #pragma unroll
      for (int i=0;i<R;i++) for (int j=0;j<S;j++) acc[i][j] += avals[i]*bvals[j];
    }

    __syncthreads();
  }

  // write back
  #pragma unroll
  for (int i=0;i<R;i++){
    int r = row0 + i;
    #pragma unroll
    for (int j=0;j<S;j++){
      int c = col0 + j;
      if (r < M && c < N) C[r*ldc + c] = acc[i][j];
    }
  }
}

// Variant B: same as A but uses float4 vectorized global loads for A and B when aligned.
// For simplicity we require BK and BN be multiples of 4 and pointers aligned; otherwise kernel still compiles but loads scalars.
template<int BM,int BN,int BK,int TX,int TY,int R,int S>
__global__ void gemm_variantB(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M,int N,int K,int lda,int ldb,int ldc)
{
  extern __shared__ float smem[];
  float* sA = smem;
  float* sB = smem + BM * BK;

  int block_m = blockIdx.y * BM;
  int block_n = blockIdx.x * BN;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row0 = block_m + ty * R;
  int col0 = block_n + tx * S;

  float acc[R][S];
  #pragma unroll
  for (int i=0;i<R;i++) for (int j=0;j<S;j++) acc[i][j]=0.0f;

  const int rows_per_threadA = BM / TY;
  const int cols_per_threadB = BN / TX;
  const int elems_per_threadA = BK / TX;
  const int elems_per_threadB = BK / TY;

  // Helpers to decide if float4 safe: require alignment and divisibility
  const bool safe4 = (sizeof(float4)==16) && ((BK % 4)==0) && ((BN % 4)==0);

  for (int kb=0; kb<K; kb+=BK){
    int start_rowA = block_m + ty * rows_per_threadA;
    int a_base_k = kb;

    // A: if safe4, load groups of 4 floats with reinterpret_cast
    #pragma unroll
    for (int r=0;r<rows_per_threadA;r++){
      int global_row = start_rowA + r;
      int sA_row = ty * rows_per_threadA + r;
      int sA_base = sA_row * BK;
      int a_glob_base = global_row*lda + a_base_k;
      if (safe4 && (a_glob_base % 4 == 0)) {
        // each thread still loads elems_per_threadA floats; group into float4 if possible
        #pragma unroll
        for (int e=0;e<elems_per_threadA;e++){
          int koff = tx * elems_per_threadA + e;
          int gk = kb + koff;
          float v = 0.0f;
          if (global_row < M && gk < K) {
            // read scalar when not multiple of 4 within row; else vector read a float4 chunk and pick lane
            // We'll attempt a friendly vector read: read 4-aligned chunk starting at (a_glob_base + koff & ~3)
            int base4 = (a_glob_base + koff) & ~3;
            float4 fv = *((const float4*)(A + base4));
            // extract scalar by lane
            int lane = (a_glob_base + koff) - base4;
            v = ((float*)&fv)[lane];
          }
          sA[sA_base + koff] = v;
        }
      } else {
        #pragma unroll
        for (int e=0;e<elems_per_threadA;e++){
          int koff = tx * elems_per_threadA + e;
          int gk = kb + koff;
          float v = 0.0f;
          if (global_row < M && gk < K) v = A[(global_row)*lda + gk];
          sA[sA_base + koff] = v;
        }
      }
    }

    // B tile loads with similar float4 attempt
    int start_colB = block_n + tx * cols_per_threadB;
    int b_base_k = kb;
    #pragma unroll
    for (int c=0;c<cols_per_threadB;c++){
      int global_col = start_colB + c;
      int sB_col = tx * cols_per_threadB + c;
      #pragma unroll
      for (int e=0;e<elems_per_threadB;e++){
        int koff = ty * elems_per_threadB + e;
        float v = 0.0f;
        int gk = kb + koff;
        if (gk < K && global_col < N) {
          v = B[gk * ldb + global_col];
        }
        sB[koff * BN + sB_col] = v;
      }
    }

    __syncthreads();

    #pragma unroll
    for (int k=0;k<BK;k++){
      float a_vals[R];
      #pragma unroll
      for (int i=0;i<R;i++){
        int srow = ty * R + i;
        a_vals[i] = sA[srow * BK + k];
      }
      float b_vals[S];
      #pragma unroll
      for (int j=0;j<S;j++){
        int scol = tx * S + j;
        b_vals[j] = sB[k * BN + scol];
      }
      #pragma unroll
      for (int i=0;i<R;i++) for (int j=0;j<S;j++) acc[i][j] += a_vals[i]*b_vals[j];
    }

    __syncthreads();
  }

  #pragma unroll
  for (int i=0;i<R;i++){
    int r = row0 + i;
    #pragma unroll
    for (int j=0;j<S;j++){
      int c = col0 + j;
      if (r < M && c < N) C[r*ldc + c] = acc[i][j];
    }
  }
}

// Variant C: double-buffered shared memory + float4-friendly loading
template<int BM,int BN,int BK,int TX,int TY,int R,int S>
__global__ void gemm_variantC(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M,int N,int K,int lda,int ldb,int ldc)
{
  // allocate double buffers: 2 * (BM*BK + BK*BN)
  extern __shared__ float smem_all[];
  float* sA0 = smem_all;                        // size BM * BK
  float* sB0 = sA0 + BM * BK;                   // size BK * BN
  float* sA1 = sB0 + BK * BN;                   // next ping
  float* sB1 = sA1 + BM * BK;

  int block_m = blockIdx.y * BM;
  int block_n = blockIdx.x * BN;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row0 = block_m + ty * R;
  int col0 = block_n + tx * S;

  float acc[R][S];
  #pragma unroll
  for (int i=0;i<R;i++) for (int j=0;j<S;j++) acc[i][j]=0.0f;

  const int rows_per_threadA = BM / TY;
  const int cols_per_threadB = BN / TX;
  const int elems_per_threadA = BK / TX;
  const int elems_per_threadB = BK / TY;

  bool use_buf0 = true;

  // prefetch first tile into buf0
  if (0 < K) {
    int kb = 0;
    // load into sA0 and sB0 similarly to before (scalar loads for simplicity)
    int start_rowA = block_m + ty * rows_per_threadA;
    #pragma unroll
    for (int r=0;r<rows_per_threadA;r++){
      int global_row = start_rowA + r;
      int sA_row = ty * rows_per_threadA + r;
      int sA_base = sA_row * BK;
      int a_glob_base = global_row*lda + kb;
      #pragma unroll
      for (int e=0;e<elems_per_threadA;e++){
        int koff = tx * elems_per_threadA + e;
        float v = 0.0f;
        int gk = kb + koff;
        if (global_row < M && gk < K) v = A[a_glob_base + koff];
        sA0[sA_base + koff] = v;
      }
    }
    int start_colB = block_n + tx * cols_per_threadB;
    #pragma unroll
    for (int c=0;c<cols_per_threadB;c++){
      int global_col = start_colB + c;
      int sB_col = tx * cols_per_threadB + c;
      #pragma unroll
      for (int e=0;e<elems_per_threadB;e++){
        int koff = ty * elems_per_threadB + e;
        float v = 0.0f;
        int gk = kb + koff;
        if (gk < K && global_col < N) v = B[gk * ldb + global_col];
        sB0[koff * BN + sB_col] = v;
      }
    }
  }
  __syncthreads();

  for (int kb=0; kb<K; kb+=BK){
    // schedule prefetch of next tile into other buffer if exists
    int next_kb = kb + BK;
    if (next_kb < K){
      // prefetch next_kb into sA1/sB1 or sA0/sB0 depending on ping-pong
      if (use_buf0){
        // write into sA1/sB1
        int start_rowA = block_m + ty * rows_per_threadA;
        #pragma unroll
        for (int r=0;r<rows_per_threadA;r++){
          int global_row = start_rowA + r;
          int sA_row = ty * rows_per_threadA + r;
          int sA_base = sA_row * BK;
          int a_glob_base = global_row*lda + next_kb;
          #pragma unroll
          for (int e=0;e<elems_per_threadA;e++){
            int koff = tx * elems_per_threadA + e;
            float v = 0.0f;
            int gk = next_kb + koff;
            if (global_row < M && gk < K) v = A[a_glob_base + koff];
            sA1[sA_base + koff] = v;
          }
        }
        int start_colB = block_n + tx * cols_per_threadB;
        #pragma unroll
        for (int c=0;c<cols_per_threadB;c++){
          int global_col = start_colB + c;
          int sB_col = tx * cols_per_threadB + c;
          #pragma unroll
          for (int e=0;e<elems_per_threadB;e++){
            int koff = ty * elems_per_threadB + e;
            float v = 0.0f;
            int gk = next_kb + koff;
            if (gk < K && global_col < N) v = B[gk * ldb + global_col];
            sB1[koff * BN + sB_col] = v;
          }
        }
      } else {
        // write into sA0/sB0
        int start_rowA = block_m + ty * rows_per_threadA;
        #pragma unroll
        for (int r=0;r<rows_per_threadA;r++){
          int global_row = start_rowA + r;
          int sA_row = ty * rows_per_threadA + r;
          int sA_base = sA_row * BK;
          int a_glob_base = global_row*lda + next_kb;
          #pragma unroll
          for (int e=0;e<elems_per_threadA;e++){
            int koff = tx * elems_per_threadA + e;
            float v = 0.0f;
            int gk = next_kb + koff;
            if (global_row < M && gk < K) v = A[a_glob_base + koff];
            sA0[sA_base + koff] = v;
          }
        }
        int start_colB = block_n + tx * cols_per_threadB;
        #pragma unroll
        for (int c=0;c<cols_per_threadB;c++){
          int global_col = start_colB + c;
          int sB_col = tx * cols_per_threadB + c;
          #pragma unroll
          for (int e=0;e<elems_per_threadB;e++){
            int koff = ty * elems_per_threadB + e;
            float v = 0.0f;
            int gk = next_kb + koff;
            if (gk < K && global_col < N) v = B[gk * ldb + global_col];
            sB0[koff * BN + sB_col] = v;
          }
        }
      }
    }
    __syncthreads();

    // pick active buffers
    float* sa = use_buf0 ? sA0 : sA1;
    float* sb = use_buf0 ? sB0 : sB1;

    // compute on active buffer
    #pragma unroll
    for (int k_inner=0;k_inner<BK;k_inner++){
      float avals[R];
      #pragma unroll
      for (int i=0;i<R;i++){
        int srow = ty * R + i;
        avals[i] = sa[srow * BK + k_inner];
      }
      float bvals[S];
      #pragma unroll
      for (int j=0;j<S;j++){
        int scol = tx * S + j;
        bvals[j] = sb[k_inner * BN + scol];
      }
      #pragma unroll
      for (int i=0;i<R;i++) for (int j=0;j<S;j++) acc[i][j] += avals[i]*bvals[j];
    }
    // flip buffer for next iteration
    use_buf0 = !use_buf0;
    __syncthreads();
  } // kb loop

  // write back
  #pragma unroll
  for (int i=0;i<R;i++){
    int r = row0 + i;
    #pragma unroll
    for (int j=0;j<S;j++){
      int c = col0 + j;
      if (r < M && c < N) C[r*ldc + c] = acc[i][j];
    }
  }
}

// Host wrapper to launch chosen variant with GemmConfig
void launch_variant(int variant,
                    const float* dA,const float* dB,float* dC,
                    int M,int N,int K,const GemmConfig &cfg,
                    cudaStream_t stream = 0)
{
  int BM = cfg.BM, BN = cfg.BN, BK = cfg.BK;
  int TX = cfg.TX, TY = cfg.TY, R = cfg.R, S = cfg.S;

  dim3 block(TX, TY);
  dim3 grid(cdiv(N, BN), cdiv(M, BM));
  int lda = K, ldb = N, ldc = N;

  size_t shmemA = (size_t)BM * BK;
  size_t shmemB = (size_t)BK * BN;
  size_t shmem = 0;
  if (variant == 0) {
    shmem = (shmemA + shmemB) * sizeof(float);
    // instantiate template
    switch (BM){
      case 64:
        gemm_variantA<64,64,16,16,16,2,2><<<grid,block,shmem,stream>>>(dA,dB,dC,M,N,K,lda,ldb,ldc); break;
      case 128:
        gemm_variantA<128,128,32,32,8,2,2><<<grid,block,shmem,stream>>>(dA,dB,dC,M,N,K,lda,ldb,ldc); break;
      default:
        // fallback generic launch (not templated) - for brevity we'll assert
        fprintf(stderr,"Unsupported BM for template instantation\n"); exit(1);
    }
  } else if (variant == 1) {
    shmem = (shmemA + shmemB) * sizeof(float);
    switch (BM){
      case 64:
        gemm_variantB<64,64,16,16,16,2,2><<<grid,block,shmem,stream>>>(dA,dB,dC,M,N,K,lda,ldb,ldc); break;
      case 128:
        gemm_variantB<128,128,32,32,8,2,2><<<grid,block,shmem,stream>>>(dA,dB,dC,M,N,K,lda,ldb,ldc); break;
      default:
        fprintf(stderr,"Unsupported BM for template instantation\n"); exit(1);
    }
  } else if (variant == 2) {
    // double buffer needs twice the shared memory (2*(BM*BK + BK*BN))
    shmem = (shmemA + shmemB) * 2 * sizeof(float);
    switch (BM){
      case 64:
        gemm_variantC<64,64,16,16,16,2,2><<<grid,block,shmem,stream>>>(dA,dB,dC,M,N,K,lda,ldb,ldc); break;
      case 128:
        gemm_variantC<128,128,32,32,8,2,2><<<grid,block,shmem,stream>>>(dA,dB,dC,M,N,K,lda,ldb,ldc); break;
      default:
        fprintf(stderr,"Unsupported BM for template instantation\n"); exit(1);
    }
  } else {
    fprintf(stderr,"Unknown variant\n"); exit(1);
  }

  CHECK_CUDA(cudaGetLastError());
}

// Minimal test driver (allocates random matrices and runs kernels)
int main(int argc,char** argv){
  // small smoke test sizes; replace with real sizes for benchmarking
  int M = 1024, N = 1024, K = 1024;
  size_t bytesA = (size_t)M * K * sizeof(float);
  size_t bytesB = (size_t)K * N * sizeof(float);
  size_t bytesC = (size_t)M * N * sizeof(float);

  float *hA = (float*)malloc(bytesA), *hB = (float*)malloc(bytesB), *hC = (float*)malloc(bytesC);
  for (size_t i=0;i<(size_t)M*K;i++) hA[i] = (float)(i%97) * 1e-3f;
  for (size_t i=0;i<(size_t)K*N;i++) hB[i] = (float)((i*7)%89) * 1e-3f;
  for (size_t i=0;i<(size_t)M*N;i++) hC[i] = 0.0f;

  float *dA, *dB, *dC;
  CHECK_CUDA(cudaMalloc(&dA, bytesA));
  CHECK_CUDA(cudaMalloc(&dB, bytesB));
  CHECK_CUDA(cudaMalloc(&dC, bytesC));
  CHECK_CUDA(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(dC, 0, bytesC));

  // Select target GPU config; we provide both default configs; choose via argv
  GemmConfig cfg;
  if (argc > 1 && strcmp(argv[1],"4090")==0) cfg = cfg_4090(); else cfg = cfg_2090();

  // Launch variant A,B,C sequentially for quick check
  printf("Launching Variant A (2x2 scalar)...\n");
  launch_variant(0, dA, dB, dC, M, N, K, cfg);
  CHECK_CUDA(cudaDeviceSynchronize());

  printf("Launching Variant B (2x2 float4-friendly)...\n");
  CHECK_CUDA(cudaMemset(dC,0,bytesC));
  launch_variant(1, dA, dB, dC, M, N, K, cfg);
  CHECK_CUDA(cudaDeviceSynchronize());

  printf("Launching Variant C (double-buffered)...\n");
  CHECK_CUDA(cudaMemset(dC,0,bytesC));
  launch_variant(2, dA, dB, dC, M, N, K, cfg);
  CHECK_CUDA(cudaDeviceSynchronize());

  // cleanup
  CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC));
  free(hA); free(hB); free(hC);

  printf("Done\n");
  return 0;
}


Basic example

// simple_thread_blocking.cu
// Compile: nvcc -O3 simple_thread_blocking.cu -o simple_tb

#include <cuda_runtime.h>
#include <cstdio>

constexpr int BM = 32;   // block tile M
constexpr int BN = 32;   // block tile N
constexpr int BK = 8;    // K tile
constexpr int TX = 8;    // threads in x
constexpr int TY = 8;    // threads in y
constexpr int R  = 2;    // per-thread micro-tile rows
constexpr int S  = 2;    // per-thread micro-tile cols

__global__ void gemm_thread_blocking(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc)
{
  extern __shared__ float smem[];        // allocate BM*BK + BK*BN floats
  float* sA = smem;                      // size BM * BK
  float* sB = smem + BM * BK;            // size BK * BN

  // block origin in C
  int block_m = blockIdx.y * BM;
  int block_n = blockIdx.x * BN;

  // thread coordinates inside the block
  int tx = threadIdx.x; // 0..TX-1 maps to micro-col groups
  int ty = threadIdx.y; // 0..TY-1 maps to micro-row groups

  // each thread's micro-tile origin in global C
  int row0 = block_m + ty * R;
  int col0 = block_n + tx * S;

  // accumulators stored in registers
  float acc[R][S];
  #pragma unroll
  for (int i=0;i<R;i++) for (int j=0;j<S;j++) acc[i][j] = 0.0f;

  // Assumptions for simplicity: BM divisible by TY, BN divisible by TX, BK divisible by TX and TY
  const int rows_per_threadA = BM / TY;   // how many A-rows each thread loads
  const int cols_per_threadB = BN / TX;   // how many B-cols each thread loads
  const int elemsA_per_thread = BK / TX;  // how many A elements along K each thread loads per row
  const int elemsB_per_thread = BK / TY;  // how many B elements along K each thread loads per col

  for (int kb = 0; kb < K; kb += BK) {
    // --- Cooperative load A_tile (BM x BK) into sA ---
    int a_row_start = block_m + ty * rows_per_threadA;
    #pragma unroll
    for (int rr = 0; rr < rows_per_threadA; ++rr) {
      int gr = a_row_start + rr;
      int sA_row = ty * rows_per_threadA + rr;           // local shared row index
      int sA_base = sA_row * BK;
      int a_base = gr * lda + kb;
      #pragma unroll
      for (int e = 0; e < elemsA_per_thread; ++e) {
        int koff = tx * elemsA_per_thread + e;           // which k-element this thread loads
        float v = 0.0f;
        int gk = kb + koff;
        if (gr < M && gk < K) v = A[a_base + koff];
        sA[sA_base + koff] = v;
      }
    }

    // --- Cooperative load B_tile (BK x BN) into sB ---
    int b_col_start = block_n + tx * cols_per_threadB;
    #pragma unroll
    for (int cc = 0; cc < cols_per_threadB; ++cc) {
      int gc = b_col_start + cc;
      int sB_col = tx * cols_per_threadB + cc;          // local shared col index
      #pragma unroll
      for (int e = 0; e < elemsB_per_thread; ++e) {
        int koff = ty * elemsB_per_thread + e;          // which k-element this thread loads
        float v = 0.0f;
        int gk = kb + koff;
        if (gk < K && gc < N) v = B[gk * ldb + gc];
        sB[koff * BN + sB_col] = v;
      }
    }

    __syncthreads();

    // --- Compute on the loaded BK tile ---
    #pragma unroll
    for (int k_inner = 0; k_inner < BK; ++k_inner) {
      // load the small column of A needed by this thread (R values)
      float a_vals[R];
      #pragma unroll
      for (int i = 0; i < R; ++i) {
        int srow = ty * R + i;
        a_vals[i] = sA[srow * BK + k_inner];
      }

      // load the small row of B needed by this thread (S values)
      float b_vals[S];
      #pragma unroll
      for (int j = 0; j < S; ++j) {
        int scol = tx * S + j;
        b_vals[j] = sB[k_inner * BN + scol];
      }

      // rank-1 update of the micro-tile
      #pragma unroll
      for (int i = 0; i < R; ++i)
        #pragma unroll
        for (int j = 0; j < S; ++j)
          acc[i][j] += a_vals[i] * b_vals[j];
    }

    __syncthreads();
  } // end kb loop

  // --- Write back accumulators to global C with bounds checks ---
  #pragma unroll
  for (int i = 0; i < R; ++i) {
    int r = row0 + i;
    #pragma unroll
    for (int j = 0; j < S; ++j) {
      int c = col0 + j;
      if (r < M && c < N) C[r * ldc + c] = acc[i][j];
    }
  }
}

// Host helper to launch
void launch_example(const float* dA,const float* dB,float* dC,int M,int N,int K){
  dim3 block(TX, TY);
  dim3 grid((N + BN - 1)/BN, (M + BM - 1)/BM);
  size_t shmem = (size_t)(BM * BK + BK * BN) * sizeof(float);
  gemm_thread_blocking<<<grid, block, shmem>>>(dA,dB,dC, M,N,K, K, N, N);
}



*/