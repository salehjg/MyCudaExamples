#pragma once

#define fixedSlicesPerBlock 8

extern float LaunchReductionR1A0(unsigned blockSize, size_t len, const float *pIn1, float *pOut1);
