//
// Created by saleh on 9/14/23.
//

#include <cassert>
#include <iostream>

#include "kernel.h"
#include "common.h"
#include "CTensor.h"
#include "CRandFiller.h"

CTensor<float> computeGold(const CTensor<float> &tnIn1, const CTensor<float> &tnIn2) {
    assert(tnIn1.GetRank() == tnIn2.GetRank() == 1);
    assert(tnIn1.GetSize() == tnIn2.GetSize());
    auto size = tnIn1.GetSize();

    CTensor<float> tnOut({size});
    for (auto idx=0; idx<size; idx++) {
        tnOut[idx] = tnIn1[idx] + tnIn2[idx];
    }
    return std::move(tnOut);
}

int main() {
    PrintHeader();

    constexpr size_t LEN = 1024*1024;

    CRandFiller<float> randFiller(-10.0f, 10.0f);
    CTensor<float> tn1({LEN}), tn2({LEN}), tn3({LEN});
    tn1.Fill(&randFiller, FillTypes::kRandom);
    tn2.Fill(&randFiller, FillTypes::kRandom);
    auto tnGold = computeGold(tn1, tn2);

    // Launch kernel
    LaunchVecAdd(32, tn1.GetSize(), tn1.GetPtrDevice(), tn2.GetPtrDevice(), tn3.GetPtrDevice());

    tn3.D2H();
    std::cout << "Matches: " << tn3.CompareHostData(tnGold, MAX_ERR_FLOAT) << std::endl;
}