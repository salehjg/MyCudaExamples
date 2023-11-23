#include "common.h"
#include <cassert>

/**
 *
 * @param shapeN
 * @param shapeM
 * @param shapeK
 * @param pInA   shapeN * shapeK
 * @param pInB   shapeK * shapeM
 * @param pOutC  shapeN * shapeM
 */
__global__
void MatMul(
        unsigned shapeN,
        unsigned shapeM,
        unsigned shapeK,
        const float *__restrict__ pInA,
        const float *__restrict__ pInB,
        float *__restrict__ pOutC) {

    assert(blockDim.z == 1);
    assert(gridDim.z == 1);
    const unsigned outI = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned outJ = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0;
    for (unsigned k = 0; k < shapeK; k++) {
        sum += pInA[outJ * shapeK + k] * pInB[k * shapeM + outI];
    }
    pOutC[outJ * shapeM + outI] = sum;
}

float LaunchMatMul(
        const dim3 &blockSize,
        unsigned shapeN,
        unsigned shapeM,
        unsigned shapeK,
        const float *pInA,
        const float *pInB,
        float *pOutC) {

    assert(blockSize.x >= 8 && blockSize.x % 32 == 0);
    assert(blockSize.y >= 8 && blockSize.y % 32 == 0);
    assert(blockSize.z == 1);
    assert(blockSize.x * blockSize.y <= 1024);

    assert(shapeM % blockSize.x == 0);
    assert(shapeN % blockSize.y == 0);

    dim3 grid;
    grid.z = 1;

    grid.x = (shapeM - 1) / blockSize.x + 1;
    grid.y = (shapeN - 1) / blockSize.y + 1;

    return TimeKernelMilliseconds([=]() {
        MatMul<<<grid, blockSize>>>(shapeN, shapeM, shapeK, pInA, pInB, pOutC);
    });
}
