#include "common.h"

__global__
void ReductionR1A0(size_t len, const float *pIn1, float *pOut1) {
    const auto tid = threadIdx.x;
    const auto gid = blockIdx.x * blockDim.x + tid;
    atomicAdd(&pOut1[0], pIn1[gid]);
}

float LaunchReductionR1A0(unsigned blockSize, size_t len, const float *pIn1, float *pOut1) {
    return TimeKernelMilliseconds([=]() {
        ReductionR1A0<<<(len-1)/blockSize+1 ,blockSize>>>(len, pIn1, pOut1);
    });
}
