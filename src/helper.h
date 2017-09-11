
#include <cuda_runtime.h>
#include <iostream>

// check Errors
#ifndef CUDA_DISABLE_ERROR_CHECKING
#define CHECK_CUDA(ans) check_cuda((ans), "", #ans, __FILE__, __LINE__)
#define CHECK_LAST(msg) check_cuda(cudaGetLastError(), msg, "CHECK_LAST", __FILE__, __LINE__)
#else
#define CHECK_CUDA(ans) {}
#define CHECK_LAST(msg) {}
#endif
inline
void check_cuda(cudaError_t code, const char* msg, const char *func, const char *file, int line) {
    if (code != cudaSuccess) {
      std::cerr << "CUDA error "
                << msg
                << ": " << cudaGetErrorString(code)
                << "(" << code
                << ") " << file
                << ":" << line
                << " in " << func
                << "\n";
      exit(1);
    }
}
