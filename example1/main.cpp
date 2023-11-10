#include <cassert>

#include "common.h"
#include "CTensor.h"
#include "CRandFiller.h"

#include "kernel.h"

CTensor<float> ComputeGold(CTensor<float> &tn1, CTensor<float> &tn2){
    assert(tn1.GetRank() == tn2.GetRank());
    assert(tn1.GetSize() == tn2.GetSize());
    assert(tn1.GetRank() == 1);

    CTensor<float> tnOut({tn1.GetSize()});
    for(size_t idx=0; idx<tnOut.GetSize(); idx++){
        tnOut[idx] = tn1[idx] + tn2[idx];
    }
    return std::move(tnOut);
}

int main(){
    PrintHeader();
    constexpr size_t LEN = 300;

    CTensor<float> tn1({LEN}), tn2({LEN}), tn3({LEN});
    CRandFiller<float> randFiller(-10.0f, +10.0f);
    tn1.Fill(&randFiller, FillTypes::kRandom);
    tn2.Fill(&randFiller, FillTypes::kRandom);

    auto tnGold = ComputeGold(tn1, tn2);

    LaunchVecAdd(256, tn3.GetSize(), tn1.GetPtrDevice(), tn2.GetPtrDevice(), tn3.GetPtrDevice());

    tn3.D2H();
    std::cout<< "Matches: "<< tn3.CompareHostData(tnGold, MAX_ERR_FLOAT)<<std::endl;
}
