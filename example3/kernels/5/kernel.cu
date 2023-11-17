#include "common.h"
#include "kernel.h"

__device__ void warpReduce(volatile float* sdata, unsigned tid) {
#pragma unroll
    for (unsigned s = 32; s > 0; s >>= 1) {
        sdata[tid] += sdata[tid + s];
    }
}

template<unsigned blockSize, unsigned slicesPerBlock>
__global__ void ReductionR1A0(size_t len, const float *__restrict__ pIn1, float *__restrict__ pOut1) {
    extern __shared__ float smem[];

    const unsigned tid = threadIdx.x;
    const size_t gid = blockIdx.x * blockDim.x + tid;

    // Phase 0. Init the output tensor
    // Comment out this phase and test the kernel with
    // `sudo compute-sanitizer --tool initcheck ./ReductionR1A0 12345`.
    if (gid == 0) {
        pOut1[0] = 0;
    }


    // Phase 1. Copy the assigned chunk to the shared memory.
    float sum = 0;
#pragma unroll
    for (unsigned i = 0; i < slicesPerBlock; i++) {
        unsigned idx = blockIdx.x * (slicesPerBlock * blockSize) + tid + i * blockSize;

        sum += pIn1[idx];

        // WARNING:
        // Float type is NOT associative, so (a+b)+c != a+(b+c)
        // So without the fence below, the order of the unrolled instructions could be varied, resulting in random float32 quantization error!
        __threadfence_block();
    }
    smem[tid] = sum;
    __syncthreads();


    // Phase 2. Tree Reduction
#pragma unroll
    for (unsigned s = blockSize / 2; s > 32; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) { // Handling s=32,16,8,4,2,1
        warpReduce(smem, tid);
    }



    // Phase 3. Accumulate the partial sum of this block with the rest.
    if (tid == 0) {
        atomicAdd(pOut1, smem[0]);
        // This is bad. For large samples, there will be huge amount of operations forcibly serialized by the atomic operation above across the grid.
        // It affects the final latency of the whole kernel, see [Thread Fence Reduction](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/threadFenceReduction/threadFenceReduction_kernel.cuh).
    }

}

float LaunchReductionR1A0(unsigned blockSize, size_t len, const float *pIn1, float *pOut1) {
    size_t grid = len / (blockSize * fixedSlicesPerBlock);
    std::cout << "gird: " << grid << std::endl;
    std::cout << "block: " << blockSize << std::endl;
    std::cout << "len: " << len << std::endl;

    if (blockSize==1024) {
        return TimeKernelMilliseconds([=]() {
            ReductionR1A0<1024, fixedSlicesPerBlock><<<grid, blockSize, blockSize * sizeof(float)>>>(len, pIn1, pOut1);
        });
    }

    if (blockSize==512) {
        return TimeKernelMilliseconds([=]() {
            ReductionR1A0<512, fixedSlicesPerBlock><<<grid, blockSize, blockSize * sizeof(float)>>>(len, pIn1, pOut1);
        });
    }

    if (blockSize==256) {
        return TimeKernelMilliseconds([=]() {
            ReductionR1A0<256, fixedSlicesPerBlock><<<grid, blockSize, blockSize * sizeof(float)>>>(len, pIn1, pOut1);
        });
    }

    if (blockSize==128) {
        return TimeKernelMilliseconds([=]() {
            ReductionR1A0<128, fixedSlicesPerBlock><<<grid, blockSize, blockSize * sizeof(float)>>>(len, pIn1, pOut1);
        });
    }

    if (blockSize==64) {
        return TimeKernelMilliseconds([=]() {
            ReductionR1A0<64, fixedSlicesPerBlock><<<grid, blockSize, blockSize * sizeof(float)>>>(len, pIn1, pOut1);
        });
    }

    if (blockSize==32) {
        return TimeKernelMilliseconds([=]() {
            ReductionR1A0<32, fixedSlicesPerBlock><<<grid, blockSize, blockSize * sizeof(float)>>>(len, pIn1, pOut1);
        });
    }

    throw std::runtime_error("The given block size is not valid for this kernel (it should be a power of two and greater than 32)");

}
