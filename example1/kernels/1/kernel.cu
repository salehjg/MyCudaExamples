#include "common.h"

__global__
void VecAdd(size_t len, const float *pIn1, const float *pIn2, float *pOut1) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<len){
       pOut1[idx] = pIn1[idx] + pIn2[idx];
    }
}

float LaunchVecAdd(unsigned blockSize, size_t len, const float *pIn1, const float *pIn2, float *pOut1) {
    return TimeKernelMilliseconds([=]() {
        VecAdd<<<(len-1)/blockSize+1 ,blockSize>>>(len, pIn1, pIn2, pOut1);
    });
}
