#include <cassert>
#include <charconv>
#include <opencv2/opencv.hpp>

#include "common.h"
#include "CTensor.h"
#include "CRandFiller.h"

#include "kernel.h"

constexpr unsigned fixedImgW = 640;
constexpr unsigned fixedImgH = 480;
constexpr unsigned fixedImgCh = 3;

__forceinline__
cv::Mat resizeImageTo640(const cv::Mat &frame) {
    cv::Mat resized_down;
    cv::resize(frame, resized_down, cv::Size(fixedImgW, fixedImgH), cv::INTER_LINEAR);
    return std::move(resized_down);
}

int main(int argc, char *argv[]) {
    PrintHeader();

    cv::VideoCapture cap;
    if (!cap.open(0))
        return 1;

    cv::Mat dummy;
    cap >> dummy;
    assert(dummy.channels() == fixedImgCh);

    const float weight[] = {-1.f,0,1.f,-2.f,0,2.f,-1.f,0,1.f};
    auto *pConstantWeightPtr = PrepareConstantWeight(weight);

    CTensor<unsigned char> tnIn({fixedImgH, fixedImgW, fixedImgCh});
    CTensor<unsigned char> tnOut({fixedImgH, fixedImgW, fixedImgCh});
    cv::Mat frameRaw, frameIn;

    for (;;) {

        cap >> frameRaw;
        if (frameRaw.empty()) break; // end of video stream
        frameIn = resizeImageTo640(frameRaw);
        tnIn.CopyHostDataFrom(frameIn.ptr());
        tnIn.H2D();

        LaunchEdgeDetection(
                fixedImgW,
                fixedImgH,
                fixedImgCh,
                tnIn.GetPtrDevice(),
                pConstantWeightPtr,
                tnOut.GetPtrDevice());

        tnOut.D2H();
        cv::Mat frameOut(fixedImgH, fixedImgW, CV_8UC3, tnOut.GetPtrHost());

        imshow("Press ESC to quit...", frameOut);
        if (cv::waitKey(1) == 27)break;
    }
}


