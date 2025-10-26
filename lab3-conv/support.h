/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#ifndef __FILEH__
#define __FILEH__

#include <sys/time.h>

typedef struct
{
    struct timeval startTime;
    struct timeval endTime;
} Timer;

// Matrix Structure declaration
typedef struct
{
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    float *elements;
} Matrix;

#define FILTER_SIZE 5
#define FILTER_RAD (FILTER_SIZE - 1)/2
#define TILE_SIZE 16 + 2*FILTER_RAD
#define BLOCK_SIZE 16

Matrix allocateMatrix(unsigned height, unsigned width);
void initMatrix(Matrix mat, bool flag);
Matrix allocateDeviceMatrix(unsigned height, unsigned width);
cudaArray* allocateDeviceArray(unsigned height, unsigned width);
cudaTextureObject_t allocateTex(cudaArray *cuArray, Matrix h_input, unsigned height, unsigned width);
void copyToDeviceMatrix(Matrix dst, Matrix src);
void copyFromDeviceMatrix(Matrix dst, Matrix src);
void verify(Matrix M, Matrix N, Matrix P);
void freeMatrix(Matrix mat);
void freeDeviceMatrix(Matrix mat);
void startTime(Timer *timer);
void stopTime(Timer *timer);
float elapsedTime(Timer timer);
void printMatrix(Matrix M);

#define FATAL(msg, ...)                                 \
    do                                                  \
    {                                                   \
        printf("[%s:%d\n%s]", __FILE__, __LINE__, msg); \
        exit(-1);                                       \
    } while (0)

#if __BYTE_ORDER != __LITTLE_ENDIAN
#error "File I/O is not implemented for this system: wrong endianness."
#endif

#endif
