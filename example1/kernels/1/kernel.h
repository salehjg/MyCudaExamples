//
// Created by saleh on 9/17/23.
//

#ifndef FARADARSCUDABASICS_KERNEL_H
#define FARADARSCUDABASICS_KERNEL_H

#include <cuda_runtime.h>

extern void LaunchVecAdd(size_t grid, size_t blockSize, const float *i1, const float *i2, float *i3);

#endif //FARADARSCUDABASICS_KERNEL_H
