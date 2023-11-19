#include "common.h"
#include "kernel.h"

template<unsigned blockSize, unsigned slicesPerBlock>
__device__ __forceinline__ void CopySlices(const float *pIn1, float *sdata, unsigned bid, unsigned tid) {
    float sum = 0;
#pragma unroll
    for (unsigned i = 0; i < slicesPerBlock; i++) {
        unsigned idx = bid * (slicesPerBlock * blockSize) + i * blockSize + tid;
        sum += pIn1[idx];
        __threadfence_block();
    }
    sdata[tid] = sum;
}

__device__ __forceinline__ void WarpReduce(/*volatile*/float *sdata, unsigned tid) {
    /*
    // Legacy warp-synchronous code, it is NO LONGER SAFE, and results in race conditions.
    // not safe since cuda>9.0  arch>volta
    // Refer to https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
    // Refer to https://developer.nvidia.com/blog/inside-volta/
    // Refer to http://www.nvidia.com/object/volta-architecture-whitepaper.html
#pragma unroll
    for (unsigned s = 32; s > 0; s >>= 1) {
        if (tid<s)
            sdata[tid] += sdata[tid + s];
    }
    */

    float v = sdata[tid];
#pragma unroll
    for (unsigned s = 32; s > 0; s >>= 1) {
        v += sdata[tid + s];

        // Forces the warp's threads (mask=all) to synchronize (includes a mem fence).
        // Without the following syncwarp(), there is no grantee that all the reads will be before all the writes to `sdata`.
        __syncwarp();

        sdata[tid] = v;
        __syncwarp();
    }
}

template<unsigned blockSize, unsigned slicesPerBlock>
__global__ void ReductionR1A0(size_t len, const float *__restrict__ pIn1, float *__restrict__ pOut1) {
    extern __shared__ float smem[];

    const unsigned tid = threadIdx.x;
    const size_t gid = blockIdx.x * blockDim.x + tid;


    // Do not try to init the output array this way.
    // There is no grantee that block0/tid0 executes before any other blocks/warps.
    // Instead, initialize the output with cudaMemSet() on the host side.
    // Use `compute-sanitizer --tool initcheck ./ExecutableName` to check for non-initialized read accesses.
    //if (gid == 0) {
    //    pOut1[0] = 0;
    //}


    // Phase 1. Copy the assigned chunk to the shared memory.
    CopySlices<blockSize, slicesPerBlock>(pIn1, smem, blockIdx.x, threadIdx.x);
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
        WarpReduce(smem, tid);
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

    if (blockSize == 1024) {
        return TimeKernelMilliseconds([=]() {
            ReductionR1A0<1024, fixedSlicesPerBlock><<<grid, blockSize, blockSize * sizeof(float)>>>(len, pIn1, pOut1);
        });
    }

    if (blockSize == 512) {
        return TimeKernelMilliseconds([=]() {
            ReductionR1A0<512, fixedSlicesPerBlock><<<grid, blockSize, blockSize * sizeof(float)>>>(len, pIn1, pOut1);
        });
    }

    if (blockSize == 256) {
        return TimeKernelMilliseconds([=]() {
            ReductionR1A0<256, fixedSlicesPerBlock><<<grid, blockSize, blockSize * sizeof(float)>>>(len, pIn1, pOut1);
        });
    }

    if (blockSize == 128) {
        return TimeKernelMilliseconds([=]() {
            ReductionR1A0<128, fixedSlicesPerBlock><<<grid, blockSize, blockSize * sizeof(float)>>>(len, pIn1, pOut1);
        });
    }

    if (blockSize == 64) {
        return TimeKernelMilliseconds([=]() {
            ReductionR1A0<64, fixedSlicesPerBlock><<<grid, blockSize, blockSize * sizeof(float)>>>(len, pIn1, pOut1);
        });
    }

    if (blockSize == 32) {
        return TimeKernelMilliseconds([=]() {
            ReductionR1A0<32, fixedSlicesPerBlock><<<grid, blockSize, blockSize * sizeof(float)>>>(len, pIn1, pOut1);
        });
    }

    throw std::runtime_error(
            "The given block size is not valid for this kernel (it should be a power of two and greater than 32)");

}
