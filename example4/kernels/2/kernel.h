#pragma once

#include <cuda_runtime.h>

constexpr unsigned fixedTileDepth = 16;
extern float LaunchMatMul(
        const dim3 &blockSize,
        unsigned shapeN,
        unsigned shapeM,
        unsigned shapeK,
        const float *pInA,
        const float *pInB,
        float *pOutC);
