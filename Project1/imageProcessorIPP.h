#ifndef IMAGEPROCESSORIPP_H
#define IMAGEPROCESSORIPP_H

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>

#include <omp.h>

#include <npp.h>
#include <nppi.h>
#include <nppi_filtering_functions.h>

#include <ipp.h>
#include <ipp/ippcore.h>
#include <ipp/ippi.h>
#include <ipp/ippcc.h>
#include <ipp/ipps.h>
#include <ipp/ippcv.h>

class ImageProcessorIPP {
public:
    ImageProcessorIPP();
    ~ImageProcessorIPP();

    cv::Mat rotate(cv::Mat& inputImage, double angle);
    cv::Mat grayScale(cv::Mat& inputImage);
    cv::Mat zoom(cv::Mat& inputImage, int newWidth, int newHeight);
    cv::Mat gaussianBlur(cv::Mat& inputImage, int kernelSize);
    cv::Mat cannyEdges(cv::Mat& inputImage);
    cv::Mat medianFilter(cv::Mat& inputImage);
    cv::Mat laplacianFilter(cv::Mat& inputImage);
    cv::Mat bilateralFilter(cv::Mat& inputImage);
    cv::Mat sobelFilter(cv::Mat& inputImage);

private:
    // 유틸리티 함수 선언
    Ipp8u* matToIpp8u(cv::Mat& mat);
};

#endif // IMAGEPROCESSORIPP_H