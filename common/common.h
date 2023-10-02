//
// Created by saleh on 9/17/23.
//

#pragma once

#include <iostream>
#include <functional>
#include <cuda_runtime.h>

constexpr float MAX_ERR_FLOAT = 0.000001f;
#define CHECK(E) if(E!=cudaError_t::cudaSuccess) std::cerr<<"CUDA API FAILED, File: "<<__FILE__<<", Line: "<< __LINE__ << ", Error: "<< cudaGetErrorString(E) << std::endl;


inline void PrintHeader() {
    std::cout << "Example Name: " << TARGETNAME << std::endl;
    std::cout << "Compiled Kernel Version: " << TARGETKERNEL << std::endl;
}

inline float TimeKernelMilliseconds(const std::function<void(void)> &kernelLaunch) {
    float time;
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start, 0));

    kernelLaunch();

    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&time, start, stop));
    std::cout << "Kernel Time (ms): "<< time << std::endl;
    return time;
}
