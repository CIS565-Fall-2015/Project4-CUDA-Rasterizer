#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) _checkCUDAErrorHelper(msg, FILENAME, __LINE__)
#define FORCEDEBUG 0

static
void _checkCUDAErrorHelper(const char *msg, const char *filename, int line) {
#if !defined(NDEBUG) || FORCEDEBUG
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (filename) {
        fprintf(stderr, " (%s:%d)", filename, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}
