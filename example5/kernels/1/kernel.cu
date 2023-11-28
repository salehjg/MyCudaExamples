#include "common.h"
#include "kernel.h"
#include <cassert>

__constant__ float weightDevice[fixedMaskWidth * fixedMaskWidth];

template<unsigned blockSize, unsigned maskWidth>
__global__
void EdgeDetection(
        unsigned width,
        unsigned height,
        unsigned channels,
        const unsigned char *__restrict__ pIn,
        const float *__restrict__ pWeight,
        unsigned char *__restrict__ pOut) {

    constexpr unsigned tileSize = blockSize - (maskWidth - 1);
    constexpr unsigned maskWidthHalf = maskWidth / 2;

    auto tx = threadIdx.x;
    auto ty = threadIdx.y;

    __shared__ float Ns[blockSize][blockSize];

    int row_o = ty + (blockIdx.y * tileSize);
    int col_o = tx + (blockIdx.x * tileSize);

    int row_i = row_o - maskWidthHalf;
    int col_i = col_o - maskWidthHalf;

    for (int color = 0; color < channels; color++) {

        if ((row_i >= 0) && (col_i >= 0) && (row_i < height) && (col_i < width)) {
            Ns[ty][tx] = pIn[(row_i * width + col_i) * 3 + color];
        } else {
            Ns[ty][tx] = 0.0f;
        }

        __syncthreads();


        float output = 0.0f;

        int i, j;
        if (ty < tileSize && tx < tileSize) {
            for (i = 0; i < maskWidth; i++) {
                for (j = 0; j < maskWidth; j++) {
                    output = output + Ns[ty + i][tx + j] * pWeight[i * maskWidth + j];

                }
            }
        }
        __syncthreads();

        if (tx < tileSize && ty < tileSize && row_o < height && col_o < width) {
            pOut[((row_o * width) + col_o) * 3 + color] = (unsigned char) output;
        }
    }
}

float LaunchEdgeDetection(
        unsigned width,
        unsigned height,
        unsigned channels,
        const unsigned char *pIn,
        const float *pConstantWeight,
        unsigned char *pOut) {

    constexpr unsigned blockSize = 8;

    dim3 block, grid;

    block.z = 1;
    grid.z = 1;

    block.x = blockSize;
    block.y = blockSize;

    grid.x = (width - 1) / block.x + 1;
    grid.y = (height - 1) / block.y + 1;

    return TimeKernelMilliseconds([=]() {
        EdgeDetection<blockSize, fixedMaskWidth>
        <<<grid, block>>>(width, height, channels, pIn, pConstantWeight, pOut);
    });
}

float* PrepareConstantWeight(const float *weightHostData) {
    CHECK(cudaMemcpyToSymbol(weightDevice, weightHostData, (fixedMaskWidth * fixedMaskWidth) * sizeof(float)));
    float *pWeightSymbolAddr;
    CHECK(cudaGetSymbolAddress((void **) &pWeightSymbolAddr, weightDevice));
    return pWeightSymbolAddr;
}
