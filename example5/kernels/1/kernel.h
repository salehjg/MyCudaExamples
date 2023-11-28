#pragma once

constexpr unsigned fixedMaskWidth = 3;

extern float LaunchEdgeDetection(
        unsigned width,
        unsigned height,
        unsigned channels,
        const unsigned char *pIn,
        const float *pConstantWeight,
        unsigned char *pOut);

extern float* PrepareConstantWeight(const float *weightHostData);
