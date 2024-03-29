#include "common.h"
#include "kernel.h"
#include <cassert>

/**
 *
 * @tparam blockSize The block should be square (2D).
 * @tparam tileDepth The W/H of the shared memory, (blockSize*tileDepth) or (tileDepth*blockSize)
 * @param shapeN
 * @param shapeM
 * @param shapeK always: shapeK >= tileDepth
 * @param pInA   shapeN * shapeK
 * @param pInB   shapeK * shapeM
 * @param pOutC  shapeN * shapeM
 */

template<unsigned blockSize, unsigned tileDepth>
__global__
void MatMul(
        unsigned shapeN,
        unsigned shapeM,
        unsigned shapeK,
        const float *__restrict__ pInA,
        const float *__restrict__ pInB,
        float *__restrict__ pOutC) {

    assert(blockDim.x == blockSize);
    assert(blockDim.y == blockSize);
    assert(blockDim.z == 1);
    assert(gridDim.z == 1);

    __shared__ float tileA[blockSize * tileDepth];
    __shared__ float tileB[tileDepth * blockSize];

    const unsigned outI = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned outJ = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned kReps = shapeK / tileDepth;
    assert(shapeK >= tileDepth); // shapeK cannot be less than tileDepth!

    float sum = 0;
    for (unsigned kTile = 0; kTile < kReps; kTile++) {
        for (unsigned equivalentTidX = threadIdx.x; equivalentTidX < tileDepth; equivalentTidX += blockSize) {
            // Our blocks are `blockSize (height) * blockSize (width)` but we are trying to copy a tile of
            // global memory of size `blockSize * tileDepth`. So we need a block-stride loop for x-axis to extend it
            // from [0...blockSize) to [0...tileDepth).
            if (equivalentTidX < tileDepth) {
                unsigned xA = kTile * tileDepth + equivalentTidX;
                unsigned yA = outJ;
                if (xA < shapeK && yA < shapeN) {
                    tileA[threadIdx.y * tileDepth + equivalentTidX] = pInA[yA * shapeK + xA];
                } else {
                    tileA[threadIdx.y * tileDepth + equivalentTidX] = 0;
                }
            }
        }
        for (unsigned equivalentTidY = threadIdx.y; equivalentTidY < tileDepth; equivalentTidY += blockSize) {
            if (equivalentTidY < tileDepth) {
                unsigned xB = outI;
                unsigned yB = kTile * tileDepth + equivalentTidY;
                if (xB < shapeM && yB < shapeK) {
                    tileB[equivalentTidY * blockSize + threadIdx.x] = pInB[yB * blockSize + xB];
                } else {
                    tileB[equivalentTidY * blockSize + threadIdx.x] = 0;
                }
            }
        }
        __syncthreads();

//#pragma unroll
        for (unsigned k = 0; k < tileDepth; k++) {
            sum += tileA[threadIdx.y * tileDepth + k] * tileB[k * blockSize + threadIdx.x];
        }
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

    assert(blockSize.x == blockSize.y);
    assert(blockSize.z == 1);

    assert(blockSize.x >= 8);
    assert(blockSize.y >= 8);
    assert(blockSize.x * blockSize.y <= 1024);

    assert(shapeM % blockSize.x == 0);
    assert(shapeN % blockSize.y == 0);

    dim3 grid;
    grid.z = 1;

    grid.x = (shapeM - 1) / blockSize.x + 1;
    grid.y = (shapeN - 1) / blockSize.y + 1;


    // Remember that the blocks are 2D and square!
    unsigned bSize = blockSize.x;
    if (bSize == 32u) {
        return TimeKernelMilliseconds([=]() {
            MatMul<32u, fixedTileDepth><<<grid, blockSize>>>(shapeN, shapeM, shapeK, pInA, pInB, pOutC);
        });
    }
    if (bSize == 16u) {
        return TimeKernelMilliseconds([=]() {
            MatMul<16u, fixedTileDepth><<<grid, blockSize>>>(shapeN, shapeM, shapeK, pInA, pInB, pOutC);
        });
    }
    if (bSize == 8u) {
        return TimeKernelMilliseconds([=]() {
            MatMul<8u, fixedTileDepth><<<grid, blockSize>>>(shapeN, shapeM, shapeK, pInA, pInB, pOutC);
        });
    }
    throw std::invalid_argument("The requested block size is not supported.");
}
