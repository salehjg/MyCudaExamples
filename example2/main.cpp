//
// Created by saleh on 9/14/23.
//

#include <cassert>
#include <iostream>
#include <cmath>

#include "cnpy.h"
#include "kernel.h"
#include "common.h"
#include "CTensor.h"
#include "CRandFiller.h"

CTensor<float> computeGold(const CTensor<float> &tnIn1, const CTensor<float> &tnIn2) {
    assert(tnIn1.GetRank() == tnIn2.GetRank() == 1);
    assert(tnIn1.GetSize() == tnIn2.GetSize());
    auto size = tnIn1.GetSize();

    CTensor<float> tnOut({size});
    for (auto idx = 0; idx < size; idx++) {
        tnOut[idx] = tnIn1[idx] + tnIn2[idx];
    }
    return std::move(tnOut);
}

CTensor<size_t> GetSliceLengths(const std::vector<size_t> &shape) {
    std::vector<size_t> lengths;
    for (int i = 0; i < shape.size(); i++) {
        size_t len = 1;
        for (int j = i + 1; j < shape.size(); j++) {
            len = len * shape[j];
        }
        lengths.push_back(len);
    }
    size_t len = lengths.size();
    CTensor<size_t> lengthsDevice({len});
    lengthsDevice.CopyHostDataFrom(lengths.data());
    lengthsDevice.H2D();
    return std::move(lengthsDevice);
}

void RunKernel(CTensor<float> &tnOut, const CTensor<float> &tn1, const CTensor<float> &tn2, BasicOperations op) {
    // For CUDA, max threads per block is 1024 (even for 3D blocks).
    // Maximum number of blocks is 65535 for each axis of 1D, 2D, or 3D grid.

    int blockSize = 256;
    int itersPerThread = std::ceil((float) tn1.GetSize() / (65535.0f * (float) blockSize));
    size_t grid = (tn1.GetSize() - 1) / blockSize + 1;
    auto sliceLengths = GetSliceLengths(tn1.GetShape());

    std::cout << "\tTensor Size: "<< tn1.GetSize() << std::endl;
    std::cout << "\tBlock Size: "<< blockSize << std::endl;
    std::cout << "\tGrid: "<< grid << std::endl;
    std::cout << "\tIters Per Thread: "<< itersPerThread << std::endl;
    std::cout << "\t";
    LaunchBasicOps(
            grid,
            blockSize,
            tn1.GetPtrDevice(),
            tn2.GetPtrDevice(),
            tnOut.GetPtrDevice(),
            tn1.GetRank(),
            tn2.GetRank(),
            tn1.GetSize(),
            itersPerThread,
            sliceLengths.GetPtrDevice(), // we cannot do `sliceLengths.data()` as it is a host pointer, not a device one.
            op);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "Use this executable as below:" << std::endl;
        std::cout << TARGETNAME << " gold_dir_path" << std::endl;
        return 1;
    }

    const std::string goldDir(argv[1]); // not going to bother using boost or std.
    const std::string exampleGoldDir = goldDir + "/example2/"; // not going to bother using boost or std.

    PrintHeader();
    constexpr auto baseDir = "";
    const std::vector<std::tuple<const std::string, const std::string, BasicOperations>> testCases = {
            std::tuple("tn2.0.npy", "tno.0.add.npy", BasicOperations::kAdd),
            std::tuple("tn2.0.npy", "tno.0.sub.npy", BasicOperations::kSub),
            std::tuple("tn2.0.npy", "tno.0.mul.npy", BasicOperations::kMul),
            std::tuple("tn2.0.npy", "tno.0.div.npy", BasicOperations::kDiv),

            std::tuple("tn2.1.npy", "tno.1.add.npy", BasicOperations::kAdd),
            std::tuple("tn2.1.npy", "tno.1.sub.npy", BasicOperations::kSub),
            std::tuple("tn2.1.npy", "tno.1.mul.npy", BasicOperations::kMul),
            std::tuple("tn2.1.npy", "tno.1.div.npy", BasicOperations::kDiv),

            std::tuple("tn2.2.npy", "tno.2.add.npy", BasicOperations::kAdd),
            std::tuple("tn2.2.npy", "tno.2.sub.npy", BasicOperations::kSub),
            std::tuple("tn2.2.npy", "tno.2.mul.npy", BasicOperations::kMul),
            std::tuple("tn2.2.npy", "tno.2.div.npy", BasicOperations::kDiv),

            std::tuple("tn2.3.npy", "tno.3.add.npy", BasicOperations::kAdd),
            std::tuple("tn2.3.npy", "tno.3.sub.npy", BasicOperations::kSub),
            std::tuple("tn2.3.npy", "tno.3.mul.npy", BasicOperations::kMul),
            std::tuple("tn2.3.npy", "tno.3.div.npy", BasicOperations::kDiv),
    };

    auto tn1 = CTensor<float>::LoadFromNumpy(exampleGoldDir + "tn1.npy");
    tn1.H2D();

    for (auto &testCase: testCases) {
        auto tn2 = CTensor<float>::LoadFromNumpy(exampleGoldDir + std::get<0>(testCase));
        tn2.H2D();
        auto tnGold = CTensor<float>::LoadFromNumpy(exampleGoldDir + std::get<1>(testCase));
        auto tnUut = CTensor<float>(tn1.GetShape());

        std::cout << "Test Case: " << std::endl;
        RunKernel(tnUut, tn1, tn2, std::get<2>(testCase));
        tnUut.D2H();
        std::cout << "\tTn1: " << std::get<0>(testCase) << std::endl;
        std::cout << "\tTn2: " << std::get<1>(testCase) << std::endl;
        std::cout << "\tMatched: " << tnUut.CompareHostData(tnGold, MAX_ERR_FLOAT) << std::endl;
    }
}