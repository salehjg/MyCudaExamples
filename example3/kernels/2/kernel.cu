#include "common.h"

__global__
void ReductionR1A0(size_t len, const float * __restrict__ pIn1, float * __restrict__ pOut1) {
    extern __shared__ float smem[];

    const size_t tid = threadIdx.x;
    const size_t gid = blockIdx.x * blockDim.x + tid;

    // Phase 0. Init the output tensor
    // Comment out this phase and test the kernel with
    // `sudo compute-sanitizer --tool initcheck ./ReductionR1A0 12345`.
    if (gid == 0){
        pOut1[0] = 0;
    }


    // Phase 1. Copy the assigned chunk to the shared memory.
    smem[tid] = pIn1[gid];
    __syncthreads();

    // Phase 2. Tree Reduction
    for(unsigned s=1; s < blockDim.x; s *= 2) {
        unsigned index = 2 * s * tid;
        if (index < blockDim.x) {
            smem[index] += smem[index + s];
        }
        __syncthreads();
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
