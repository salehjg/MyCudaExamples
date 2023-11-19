#include "common.h"

__device__ void warpReduce(volatile float* sdata, unsigned tid) {
#pragma unroll
    for (unsigned s = 32; s > 0; s >>= 1) {
        sdata[tid] += sdata[tid + s];
    }
}

__global__
void ReductionR1A0(size_t len, const float * __restrict__ pIn1, float * __restrict__ pOut1) {
    extern __shared__ float smem[];

    const size_t tid = threadIdx.x;
    const size_t gid = blockIdx.x * blockDim.x + tid;

    // Phase 1. Copy the assigned chunk to the shared memory.
    smem[tid] = pIn1[gid];
    __syncthreads();

    // Phase 2. Tree Reduction
    for(unsigned s=blockDim.x/2; s>32; s >>=1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    if (tid<32){ // Handling s=32,16,8,4,2,1
        warpReduce(smem, tid);
    }

    // Phase 3. Accumulate the partial sum of this block with the rest.
    if (tid == 0) {
        atomicAdd(pOut1, smem[0]);
    }
}

float LaunchReductionR1A0(unsigned blockSize, size_t len, const float *pIn1, float *pOut1) {
    size_t grid = (len-1)/blockSize+1;
    std::cout<<"gird: "<< grid << std::endl;
    std::cout<<"block: "<< blockSize << std::endl;
    std::cout<<"len: "<< len << std::endl;

    return TimeKernelMilliseconds([=]() {
        ReductionR1A0<<<grid ,blockSize, blockSize*sizeof(float)>>>(len, pIn1, pOut1);
    });
}
