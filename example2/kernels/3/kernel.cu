//
// Created by saleh on 9/14/23.
//

#include <vector>
#include <cassert>
#include "common.h"
#include "kernel.h"

constexpr int UNROLLED_RANK1 = 4;
__constant__ size_t sliceLensDevice[UNROLLED_RANK1];

template<BasicOperations op>
inline __device__ float PerformOperation(float in1, float in2) {}

template<>
inline __device__ float PerformOperation<BasicOperations::kAdd>(float in1, float in2) {
    return in1 + in2;
}

template<>
inline __device__ float PerformOperation<BasicOperations::kSub>(float in1, float in2) {
    return in1 - in2;
}

template<>
inline __device__ float PerformOperation<BasicOperations::kMul>(float in1, float in2) {
    return in1 * in2;
}

template<>
inline __device__ float PerformOperation<BasicOperations::kDiv>(float in1, float in2) {
    return in1 / in2;
}

/**
 * @brief Computes the dimension index for the given axis.
 *
 * @param idx The assigned flat index for the current thread.
 * @param sliceLens An array of
 *                  {(shape1[1] * shape1[2] * shape1[3]), (shape1[2] * shape1[3]), (shape1[3]), 1}
 *                  for a tensor of rank 4 for example.
 * @return
 */

template<int rank1, int axis>
inline __device__ size_t ComputeAxisIndex(size_t idx, const size_t *sliceLens, size_t *indices) {
    // assert() only works in the main kernel function (tagged with `__global__`).
    // here, since our `__device__` function is inline, we can use assert() without problems.
    static_assert(axis < rank1);

    if constexpr (axis == 0) {
        indices[axis] = idx / sliceLens[axis];
    } else {
        indices[axis] = (idx % sliceLens[axis - 1]) / sliceLens[axis];
    }
}


template<int start, int end, typename F, typename... Args>
void __device__ CompileTimeFor(F f, size_t idx, const size_t *sliceLens, size_t *indices) {
    if constexpr (start < end) {
        f.template operator()<start>(idx, sliceLens, indices);
        CompileTimeFor<start + 1, end>(f, idx, sliceLens, indices);
    }
}

template<int rank1>
struct ComputeAxisIndexWrapper {
    template<int I>
    inline __device__ size_t operator()(size_t idx, const size_t *sliceLens, size_t *indices) {
        return ComputeAxisIndex<rank1, I>(idx, sliceLens, indices);
    }
};

/**
 * @brief Element-wise Basic Operations' Kernel. Could accept input tensor 1 of any rank.
 *
 * @param pIn1 The input tensor 1.
 * @param pIn2 The input tensor 2.
 * @param pOut1 The output tensor.
 * @param rank2 The rank of the input tensor 2. It should be equal or less that rank1.
 * @param sizeIn1 The size of the input tensor 1.
 * @param sliceLens An array of length `rank1` containing
 *                  {(shape1[1] * shape1[2] * shape1[3]), (shape1[2] * shape1[3]), (shape1[3]), 1}
 *                  for a tensor of rank 4 for example.
 */
template<size_t iterPerThread, int rank1, int rank2, BasicOperations op>
__global__ void BasicOps(
        const float *__restrict__ pIn1,
        const float *__restrict__ pIn2,
        float *__restrict__ pOut1,
        size_t sizeIn1,
        const size_t *sliceLens) {

    size_t idx, idxS2;
    size_t indices[rank1];

#pragma unroll
    for (size_t i = 0; i < iterPerThread; i++) {
        idx = (blockIdx.x * blockDim.x + threadIdx.x) * iterPerThread + i;

        // If iterPerThread>1, there might be an `idx` that is assigned to an element outside the tensor boundaries.
        if (idx >= sizeIn1) continue;

        CompileTimeFor<0, rank1>(ComputeAxisIndexWrapper<rank1>(), idx, sliceLens, indices);

        idxS2 = 0;
#pragma unroll
        for (int axis = rank1 - rank2; axis < rank1; axis++) {
            idxS2 += indices[axis] * sliceLens[axis];
        }
        pOut1[idx] = PerformOperation<op>(pIn1[idx], pIn2[idxS2]);
    }
}

/**
 *
 * @param grid The 1D grid. Calculated based on two given parameters: 1) inputTn1's size, 2) iterPerThread
 *             So, if the size of the tensor is larger than (max_grid_size * block_size), iterPerThread should be increased.
 * @param blockSize The 1D block size.
 * @param pIn1 The input tensor 1.
 * @param pIn2 The input tensor 2.
 * @param pOut1 The output tensor.
 * @param rank1 The rank of the input tensor 1.
 * @param rank2 The rank of the input tensor 2. It should be equal or less that rank1.
 * @param sizeIn1 The size of the input tensor 1.
 * @param iterPerThread Number of output tensor's elements assigned per thread.
 * @param sliceLens An array of length `rank1` containing
 *                  {(shape1[1] * shape1[2] * shape1[3]), (shape1[2] * shape1[3]), (shape1[3]), 1}
 *                  for a tensor of rank 4 for example.
 * @param op The element-wise operation that is to be done.
 * @return
 */
float LaunchBasicOps(
        unsigned grid,
        unsigned blockSize,
        const float *pIn1,
        const float *pIn2,
        float *pOut1,
        int rank1,
        int rank2,
        size_t sizeIn1,
        size_t iterPerThread,
        const size_t *sliceLensHostData,
        BasicOperations op) {

    assert(rank2 <= rank1);
    assert(rank1 < UNROLLED_RANK1);

    CHECK(cudaMemcpyToSymbol(sliceLensDevice, sliceLensHostData, rank1 * sizeof(size_t)));

    size_t *sliceLensDeviceConstAddr;
    CHECK(cudaGetSymbolAddress((void **) &sliceLensDeviceConstAddr, sliceLensDevice));

    return TimeKernelMilliseconds([=]() {
        if (iterPerThread == 1 and rank1 == 4 and rank2 == 4 and op == BasicOperations::kAdd) {
            BasicOps<1, 4, 4, BasicOperations::kAdd><<<grid, blockSize>>>(pIn1, pIn2, pOut1, sizeIn1, sliceLensDeviceConstAddr);
        }
        if (iterPerThread == 1 and rank1 == 4 and rank2 == 4 and op == BasicOperations::kSub) {
            BasicOps<1, 4, 4, BasicOperations::kSub><<<grid, blockSize>>>(pIn1, pIn2, pOut1, sizeIn1, sliceLensDeviceConstAddr);
        }
        if (iterPerThread == 1 and rank1 == 4 and rank2 == 4 and op == BasicOperations::kMul) {
            BasicOps<1, 4, 4, BasicOperations::kMul><<<grid, blockSize>>>(pIn1, pIn2, pOut1, sizeIn1, sliceLensDeviceConstAddr);
        }
        if (iterPerThread == 1 and rank1 == 4 and rank2 == 4 and op == BasicOperations::kDiv) {
            BasicOps<1, 4, 4, BasicOperations::kDiv><<<grid, blockSize>>>(pIn1, pIn2, pOut1, sizeIn1, sliceLensDeviceConstAddr);
        }


        if (iterPerThread == 1 and rank1 == 4 and rank2 == 3 and op == BasicOperations::kAdd) {
            BasicOps<1, 4, 3, BasicOperations::kAdd><<<grid, blockSize>>>(pIn1, pIn2, pOut1, sizeIn1, sliceLensDeviceConstAddr);
        }
        if (iterPerThread == 1 and rank1 == 4 and rank2 == 3 and op == BasicOperations::kSub) {
            BasicOps<1, 4, 3, BasicOperations::kSub><<<grid, blockSize>>>(pIn1, pIn2, pOut1, sizeIn1, sliceLensDeviceConstAddr);
        }
        if (iterPerThread == 1 and rank1 == 4 and rank2 == 3 and op == BasicOperations::kMul) {
            BasicOps<1, 4, 3, BasicOperations::kMul><<<grid, blockSize>>>(pIn1, pIn2, pOut1, sizeIn1, sliceLensDeviceConstAddr);
        }
        if (iterPerThread == 1 and rank1 == 4 and rank2 == 3 and op == BasicOperations::kDiv) {
            BasicOps<1, 4, 3, BasicOperations::kDiv><<<grid, blockSize>>>(pIn1, pIn2, pOut1, sizeIn1, sliceLensDeviceConstAddr);
        }


        if (iterPerThread == 1 and rank1 == 4 and rank2 == 2 and op == BasicOperations::kAdd) {
            BasicOps<1, 4, 2, BasicOperations::kAdd><<<grid, blockSize>>>(pIn1, pIn2, pOut1, sizeIn1, sliceLensDeviceConstAddr);
        }
        if (iterPerThread == 1 and rank1 == 4 and rank2 == 2 and op == BasicOperations::kSub) {
            BasicOps<1, 4, 2, BasicOperations::kSub><<<grid, blockSize>>>(pIn1, pIn2, pOut1, sizeIn1, sliceLensDeviceConstAddr);
        }
        if (iterPerThread == 1 and rank1 == 4 and rank2 == 2 and op == BasicOperations::kMul) {
            BasicOps<1, 4, 2, BasicOperations::kMul><<<grid, blockSize>>>(pIn1, pIn2, pOut1, sizeIn1, sliceLensDeviceConstAddr);
        }
        if (iterPerThread == 1 and rank1 == 4 and rank2 == 2 and op == BasicOperations::kDiv) {
            BasicOps<1, 4, 2, BasicOperations::kDiv><<<grid, blockSize>>>(pIn1, pIn2, pOut1, sizeIn1, sliceLensDeviceConstAddr);
        }


        if (iterPerThread == 1 and rank1 == 4 and rank2 == 1 and op == BasicOperations::kAdd) {
            BasicOps<1, 4, 1, BasicOperations::kAdd><<<grid, blockSize>>>(pIn1, pIn2, pOut1, sizeIn1, sliceLensDeviceConstAddr);
        }
        if (iterPerThread == 1 and rank1 == 4 and rank2 == 1 and op == BasicOperations::kSub) {
            BasicOps<1, 4, 1, BasicOperations::kSub><<<grid, blockSize>>>(pIn1, pIn2, pOut1, sizeIn1, sliceLensDeviceConstAddr);
        }
        if (iterPerThread == 1 and rank1 == 4 and rank2 == 1 and op == BasicOperations::kMul) {
            BasicOps<1, 4, 1, BasicOperations::kMul><<<grid, blockSize>>>(pIn1, pIn2, pOut1, sizeIn1, sliceLensDeviceConstAddr);
        }
        if (iterPerThread == 1 and rank1 == 4 and rank2 == 1 and op == BasicOperations::kDiv) {
            BasicOps<1, 4, 1, BasicOperations::kDiv><<<grid, blockSize>>>(pIn1, pIn2, pOut1, sizeIn1, sliceLensDeviceConstAddr);
        }
    });
}
