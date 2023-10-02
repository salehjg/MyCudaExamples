//
// Created by saleh on 9/17/23.
//

#pragma once

#include <iostream>

enum class BasicOperations{
    kAdd,
    kSub,
    kMul,
    kDiv
};

extern float LaunchBasicOps(
        unsigned grid,
        unsigned blockSize,
        const float *pIn1,
        const float *pIn2,
        float *pOut1,
        int rank1,
        int rank2,
        size_t sizeIn1,
        size_t iterPerThread,
        const size_t *sliceLens,
        BasicOperations op);
