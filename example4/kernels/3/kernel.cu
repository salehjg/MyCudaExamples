#include "common.h"
#include "kernel.h"
#include <cassert>

/**
 *
 * @tparam blockSize The block should be square (2D).
 * @tparam tileDepth The W/H of the shared memory, (blockSize*tileDepth) or (tileDepth*blockSize)
 * @tparam r_thread Each thread will calculate `r_thread*r_thread` elements of pOutC.
 * @param shapeN
 * @param shapeM
 * @param shapeK always: shapeK >= tileDepth
 * @param pInA   shapeN * shapeK
 * @param pInB   shapeK * shapeM
 * @param pOutC  shapeN * shapeM
 */

template<unsigned blockSize, unsigned tileDepth, unsigned r_thread>
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
    assert(shapeN % (blockSize * r_thread) == 0);
    assert(shapeM % (blockSize * r_thread) == 0);

    __shared__ float tileA[blockSize * r_thread][tileDepth];
    __shared__ float tileB[tileDepth][blockSize * r_thread];
    float sum[r_thread * r_thread] = {0}; // registers

    const unsigned outI = (blockIdx.x * blockDim.x + threadIdx.x) * r_thread;
    const unsigned outJ = (blockIdx.y * blockDim.y + threadIdx.y) * r_thread;

    unsigned kReps = shapeK / tileDepth;
    assert(shapeK >= tileDepth); // shapeK cannot be less than tileDepth!


    for (unsigned kTile = 0; kTile < kReps; kTile++) {
        for (unsigned equivalentTidX = threadIdx.x; equivalentTidX < tileDepth; equivalentTidX += blockSize) {
            // Our blocks are `blockSize (height) * blockSize (width)` but we are trying to copy a tile of
            // global memory of size `blockSize * tileDepth`. So we need a block-stride loop for x-axis to extend it
            // from [0...blockSize) to [0...tileDepth).
            if (equivalentTidX < tileDepth) {
                unsigned xA = kTile * tileDepth + equivalentTidX;
#pragma unroll
                for (unsigned j = 0; j < r_thread; j++) {
                    unsigned yA = outJ + j;
                    if (xA < shapeK && yA < shapeN) {
                        tileA[threadIdx.y + j][equivalentTidX] = pInA[yA * shapeK + xA];
                    } else {
                        tileA[threadIdx.y + j][equivalentTidX] = 0;
                    }
                }
            }
        }
        for (unsigned equivalentTidY = threadIdx.y; equivalentTidY < tileDepth; equivalentTidY += blockSize) {
            if (equivalentTidY < tileDepth) {
                unsigned yB = kTile * tileDepth + equivalentTidY;
#pragma unroll
                for (unsigned i = 0; i < r_thread; i++) {
                    unsigned xB = outI + i;
                    if (xB < shapeM && yB < shapeK) {
                        tileB[equivalentTidY][threadIdx.x + i] = pInB[yB * blockSize + xB];
                    } else {
                        tileB[equivalentTidY][threadIdx.x + i] = 0;
                    }
                }
            }
        }
        __syncthreads();


#pragma unroll
        for (unsigned i = 0; i < r_thread; i++) {
            float tempB[tileDepth]; // registers

            for (unsigned j = 0; j < tileDepth; j++) {
                tempB[j] = tileB[j][threadIdx.x + i];
            }

            for (unsigned j = 0; j < r_thread; j++) {
                for (unsigned k = 0; k < tileDepth; k++) {
                    sum[j * r_thread + i] += tileA[threadIdx.y][k] * tempB[k];
                }
            }
        }

    }

#pragma unroll
    for (unsigned i = 0; i < r_thread; i++) {
        for (unsigned j = 0; j < r_thread; j++) {
            pOutC[(outJ + j) * shapeM + (outI + i)] = sum[j * r_thread + i];
        }
    }
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

    assert(shapeN % (blockSize.y * fixedR) == 0);
    assert(shapeM % (blockSize.x * fixedR) == 0);

    dim3 grid;
    grid.z = 1;

    grid.x = shapeM / blockSize.x / fixedR;
    grid.y = shapeN / blockSize.y / fixedR;


    // Remember that the blocks are 2D and square!
    unsigned bSize = blockSize.x;
    if (bSize == 32u) {
        return TimeKernelMilliseconds([=]() {
            MatMul<32u, fixedTileDepth, fixedR><<<grid, blockSize>>>(shapeN, shapeM, shapeK, pInA, pInB, pOutC);
        });
    }
    if (bSize == 16u) {
        return TimeKernelMilliseconds([=]() {
            MatMul<16u, fixedTileDepth, fixedR><<<grid, blockSize>>>(shapeN, shapeM, shapeK, pInA, pInB, pOutC);
        });
    }
    if (bSize == 8u) {
        return TimeKernelMilliseconds([=]() {
            MatMul<8u, fixedTileDepth, fixedR><<<grid, blockSize>>>(shapeN, shapeM, shapeK, pInA, pInB, pOutC);
        });
    }
    throw std::invalid_argument("The requested block size is not supported.");
}
