//
// Created by saleh on 9/14/23.
//

#include "common.h"
#include "kernel.h"

int main() {
    CTensor<float> tn1({1024}), tn2({1024}), tn3({1024});
    tn1.Fill(FillTypes::kRandom);
    tn2.Fill(FillTypes::kRandom);

    // Launch kernel
    LaunchVecAdd(ceil(tn3.GetSize() / 256), 256, tn1.GetPtrDevice(), tn2.GetPtrDevice(), tn3.GetPtrDevice());
}