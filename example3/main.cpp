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

    double sum = 0; // for large partial sums, `float` will not be adequate.
    for (size_t idx = 0; idx < tn1.GetSize(); idx++) {
        if (tn1[idx] != 1.0f) {
            int a = 1;
            a++;
        }
        sum += tn1[idx];
    }
    tnOut[0] = (float)sum;

    return std::move(tnOut);;
}

size_t GetPaddedLen(unsigned blockSize, size_t wordCount, unsigned slicesPerBlock=fixedSlicesPerBlock) {
    assert(wordCount > 0);
    return ((wordCount - 1) / (slicesPerBlock*blockSize) + 1) * blockSize*slicesPerBlock;
}

template<typename T>
void FillPadRangeWithZero(size_t wordCount, CTensor<T> &tn1) {
    for (size_t i = wordCount; i < tn1.GetSize(); i++) {
        tn1[i] = 0;
    }
    if (wordCount != tn1.GetSize()) {
        tn1.H2D();
    }

}

int main(int argc, char *argv[]) {
    PrintHeader();

    if (argc != 2) {
        std::cout << "Use this executable as below:" << std::endl;
        std::cout << TARGETNAME << " <Sample Length (float words)>" << std::endl;
        return 1;
    }

    constexpr unsigned BLOCKSIZE = 512;
    unsigned LEN;
    const std::string lenStr(argv[1]);
    auto [ptr, ec] = std::from_chars(lenStr.data(), lenStr.data() + lenStr.size(), LEN);
    assert(ec == std::errc{});

    assert((BLOCKSIZE > 0 && ((BLOCKSIZE & (BLOCKSIZE - 1)) == 0))); // Check to make sure BLOCKSIZE is a power of 2.
    CTensor<float> tn1({GetPaddedLen(BLOCKSIZE, LEN)}), tn2({1});
    CRandFiller<float> randFiller(-1000000.0f, +100000.0f);

    // Do not use FillTypes::kIncr for large samples, it will overflow.
    tn1.Fill(&randFiller, FillTypes::kConstant1);
    FillPadRangeWithZero(LEN, tn1);

    // Init the output tensor (tn2)
    // This:
    tn2[0] = 0; tn2.H2D();
    //Or
    //CHECK(cudaMemset(tn2.GetPtrDevice(), 0, tn2.GetSizeBytes()));

    auto tnGold = ComputeGoldR1A0(tn1);

    LaunchReductionR1A0(BLOCKSIZE, tn1.GetSize(), tn1.GetPtrDevice(), tn2.GetPtrDevice());

    tn2.D2H();
    std::cout << "Padded Sample Size (MB): " << (float) tn1.GetSizeBytes() / 1024.0f / 1024.0f << std::endl;
    std::cout << "Padded Words: " << tn1.GetSize() - LEN << std::endl;
    std::cout << "Gold: " << tnGold[0] << std::endl;
    std::cout << "UUT: " << tn2[0] << std::endl;
    float v1 = tnGold[0];
    float v2 = tn2[0];
    std::cout << "Gold - UUT: " << v1 - v2 << std::endl;
}
