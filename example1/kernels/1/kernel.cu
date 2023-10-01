//
// Created by saleh on 9/14/23.
//

#include "common.h"

__global__
void VecAdd(const float *pIn1, const float *pIn2, float *pOut1) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    pOut1[idx] = pIn1[idx] + pIn2[idx];
}

// This function cannot be a template.
// Since different compilers are used to compile *.cu and *.cpp files.
// But the kernel itself could be templated and called in this function's body.
float LaunchVecAdd(unsigned blockSize, size_t len, const float *i1, const float *i2, float *i3) {
    return TimeKernelMilliseconds([=]() {
        VecAdd<<<(len - 1) / blockSize + 1, blockSize>>>(i1, i2, i3);
    });
}
