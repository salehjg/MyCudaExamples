//
// Created by saleh on 9/14/23.
//

#include <vector>
#include <cassert>
#include "common.h"
#include "kernel.h"

inline __device__ float PerformOperation(BasicOperations op, float in1, float in2) {
    assert(op == BasicOperations::kAdd || op == BasicOperations::kSub || op == BasicOperations::kMul ||
           op == BasicOperations::kDiv);
    return
            op == BasicOperations::kAdd ? in1 + in2 :
            op == BasicOperations::kSub ? in1 - in2 :
            op == BasicOperations::kMul ? in1 * in2 :
            op == BasicOperations::kDiv ? in1 / in2 :
            0;
}

/**
 * @brief Computes the dimension index for the given axis.
 *
 * @param axis The axis for which the dimension index should be computed.
 * @param idx The assigned flat index for the current thread.
 * @param sliceLens An array of
 *                  {(shape1[1] * shape1[2] * shape1[3]), (shape1[2] * shape1[3]), (shape1[3]), 1}
 *                  for a tensor of rank 4 for example.
 * @param rank1 The rank of the input tensor 1.
 * @return
 */
inline __device__ size_t ComputeAxisIndex(int axis, size_t idx, const size_t *sliceLens, int rank1) {
    // assert() only works in the main kernel function (tagged with `__global__`).
    // here, since our `__device__` function is inline, we can use assert() without problems.
    assert(axis < rank1);

    if (axis == 0) {
        return idx / sliceLens[axis];
    } else {
        return (idx % sliceLens[axis - 1]) / sliceLens[axis];
    }
}

/**
 * @brief Element-wise Basic Operations' Kernel. Could accept input tensor 1 of any rank.
 *
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
 */
__global__
void BasicOps(
        const float *__restrict__ pIn1,
        const float *__restrict__ pIn2,
        float *__restrict__ pOut1,
        int rank1,
        int rank2,
        size_t sizeIn1,
        size_t iterPerThread,
        const size_t *sliceLens,
        BasicOperations op) {

    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx, idxS2;

    //auto indices = new size_t[rank1];
    assert(rank1 < 5);
    size_t indices[5];

    for (size_t i = 0; i < iterPerThread; i++) {
        idx = gid * iterPerThread + i;

        // If iterPerThread>1, there might be an `idx` that is assigned to an element outside the tensor boundaries.
        if (idx >= sizeIn1) continue;

        for (int axis = 0; axis < rank1; axis++) {
            indices[axis] = ComputeAxisIndex(axis, idx, sliceLens, rank1);
        }

        idxS2 = 0;
        for (int axis = rank1 - rank2; axis < rank1; axis++) {
            idxS2 += indices[axis] * sliceLens[axis];
        }
        pOut1[idx] = PerformOperation(op, pIn1[idx], pIn2[idxS2]);
    }

    // VERY IMPORTANT: any local memory allocated with `new[]` in the kernel should be `delete[]`ed at the end of the kernel.
    // These dynamically allocated local arrays stay on the global memory even after the kernel's excuted.
    // For example if you comment out the next line below, the kernel will crash for input tensor with large shapes.
    //delete[] indices;
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
        const size_t *sliceLensHost,
        BasicOperations op) {

    assert(rank2 <= rank1);

    size_t *sliceLensDevice;
    CHECK(cudaMalloc((void **) &sliceLensDevice, rank1 * sizeof(size_t)));
    CHECK(cudaMemcpy(sliceLensDevice, sliceLensHost, rank1 * sizeof(size_t), cudaMemcpyHostToDevice));

    auto t = TimeKernelMilliseconds([=]() {
        BasicOps<<<grid, blockSize>>>(pIn1, pIn2, pOut1, rank1, rank2, sizeIn1, iterPerThread, sliceLensDevice, op);
    });

    CHECK(cudaFree(sliceLensDevice));
    return t;
}
