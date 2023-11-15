#include <cassert>
#include <charconv>

#include "common.h"
#include "CTensor.h"
#include "CRandFiller.h"

#include "kernel.h"

template<typename T>
CTensor<T> ComputeGoldR1A0(CTensor<T> &tn1) {
    assert(tn1.GetRank() == 1);
    CTensor<float> tnOut({1});
    tnOut[0] = 0;
    for (size_t idx = 0; idx < tn1.GetSize(); idx++) {
        tnOut[0] += tn1[idx];
    }
    return std::move(tnOut);;
}

size_t GetPaddedLen(unsigned blockSize, size_t wordCount) {
    assert(wordCount > 0);
    return ((wordCount - 1) / blockSize + 1) * blockSize;
}

template<typename T>
void FillPadRangeWithZero(size_t wordCount, CTensor<T> &tn1) {
    for (size_t i = wordCount; i < tn1.GetSize(); i++) {
        tn1[i] = 0;
    }
    if (wordCount != tn1.GetSize()) tn1.H2D();
}

int main(int argc, char *argv[]) {
    PrintHeader();

    if (argc != 2) {
        std::cout << "Use this executable as below:" << std::endl;
        std::cout << TARGETNAME << " <Sample Length (float words)>" << std::endl;
        return 1;
    }

    const unsigned BLOCKSIZE = 512;
    unsigned LEN;
    const std::string lenStr(argv[1]);
    auto [ptr, ec] = std::from_chars(lenStr.data(), lenStr.data()+lenStr.size(), LEN);
    assert(ec == std::errc{});

    assert((BLOCKSIZE > 0 && ((BLOCKSIZE & (BLOCKSIZE - 1)) == 0))); // Check to make sure BLOCKSIZE is a power of 2.
    CTensor<float> tn1({GetPaddedLen(BLOCKSIZE, LEN)}), tn2({1});
    CRandFiller<float> randFiller(-1000000.0f, +100000.0f);

    // Do not use FillTypes::kIncr for large samples, it will overflow.
    tn1.Fill(&randFiller, FillTypes::kConstant1);
    FillPadRangeWithZero(LEN, tn1);

    auto tnGold = ComputeGoldR1A0(tn1);

    LaunchReductionR1A0(256, tn1.GetSize(), tn1.GetPtrDevice(), tn2.GetPtrDevice());

    tn2.D2H();
    std::cout << "Padded Sample Size (MB): " << (float) tn1.GetSizeBytes() / 1024.0f / 1024.0f << std::endl;
    std::cout << "Padded Words: " << tn1.GetSize() - LEN << std::endl;
    std::cout << "Gold - UUT: " << tnGold[0] - tn2[0] << std::endl;
}
