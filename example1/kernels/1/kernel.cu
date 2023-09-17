//
// Created by saleh on 9/14/23.
//

__global__
void VecAdd(const float *pIn1, const float *pIn2, float *pOut1) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    pOut1[idx] = pIn1[idx] + pIn2[idx];
}

void LaunchVecAdd(size_t grid, size_t blockSize, const float *i1, const float *i2, float *i3) {
    VecAdd<<<grid, blockSize>>>(i1, i2, i3);
}
