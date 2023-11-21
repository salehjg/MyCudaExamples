#include <cassert>
#include <charconv>

#include "common.h"
#include "CTensor.h"
#include "CRandFiller.h"

#include "kernel.h"

template<typename T>
CTensor<T> ComputeGold(const CTensor<T> &tnA, const CTensor<T> &tnB) {
    assert(tnA.GetRank() == 2);
    assert(tnB.GetRank() == 2);
    assert(tnA.GetShape()[1] == tnB.GetShape()[0]);
    const unsigned N = tnA.GetShape()[0], M = tnB.GetShape()[1], K = tnA.GetShape()[1];
    CTensor<T> tnC({N, M});
    for (unsigned j = 0; j < N; j++) {
        for (unsigned i = 0; i < M; i++) {
            T sum = 0;
            for (unsigned c = 0; c < K; c++) { // common
                sum += tnA[j * K + c] * tnB[c * M + i];
            }
            tnC[j * M + i] = sum;
        }
    }
    return std::move(tnC);
}

int main(int argc, char *argv[]) {
    PrintHeader();

    if (argc != 4) {
        std::cout << "Use this executable as below:" << std::endl;
        std::cout << TARGETNAME << " <N> <M> <K>" << std::endl;
        std::cout << "\tN: The number of rows in the output matrix." << std::endl;
        std::cout << "\tM: The number of columns in the output matrix." << std::endl;
        std::cout << "\tK: The number of rows/columns in the common axis." << std::endl;
        return 1;
    }

    constexpr unsigned BLOCKSIZE = 512;
    unsigned shapeN, shapeM, shapeK;

    for (int i = 1; i < 4; i++) {
        const std::string lenStr(argv[i]);
        unsigned *val;
        val = i == 1 ? &shapeN : i == 2 ? &shapeM : &shapeK;
        auto [ptr, ec] = std::from_chars(lenStr.data(), lenStr.data() + lenStr.size(), *val);
        assert(ec == std::errc{});
    }

    // Check to make sure BLOCKSIZE is a power of 2.
    assert((BLOCKSIZE > 0 && ((BLOCKSIZE & (BLOCKSIZE - 1)) == 0)));

    CTensor<float> tnA({shapeN, shapeK}), tnB({shapeK, shapeM}), tnC({shapeN, shapeM});
    CRandFiller<float> randFiller(-11.0f, +11.0f);

    tnA.Fill(&randFiller, FillTypes::kConstant1);
    tnB.Fill(&randFiller, FillTypes::kConstant1);
    tnC.Fill(&randFiller, FillTypes::kConstant0);

    auto tnGold = ComputeGold(tnA, tnB);

    LaunchMatMul(
            {8, 8, 1},
            shapeN, shapeM, shapeK,
            tnA.GetPtrDevice(),
            tnB.GetPtrDevice(),
            tnC.GetPtrDevice()
    );

    tnC.D2H();
    std::cout << "Matched: " << tnC.CompareHostData(tnGold, MAX_ERR_FLOAT) << std::endl;
}


