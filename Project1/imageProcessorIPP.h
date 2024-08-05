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

#include "AbstractImageProcessor.h"
#include "ImageProcessingLib.h"

class ImageProcessorIPP : public AbstractImageProcessor {
public:
    ImageProcessorIPP();
    ~ImageProcessorIPP();

    cv::Mat rotate(cv::Mat& inputImage, bool isRight) override;
    cv::Mat grayScale(cv::Mat& inputImage) override;
    cv::Mat zoom(cv::Mat& inputImage, int newWidth, int newHeight) override;
    cv::Mat gaussianBlur(cv::Mat& inputImage, int kernelSize) override;
    cv::Mat cannyEdges(cv::Mat& inputImage) override;
    cv::Mat medianFilter(cv::Mat& inputImage) override;
    cv::Mat laplacianFilter(cv::Mat& inputImage) override;
    cv::Mat bilateralFilter(cv::Mat& inputImage) override;
    cv::Mat sobelFilter(cv::Mat& inputImage) override;

private:
    std::string getClassName() const override;
};

#endif // IMAGEPROCESSORIPP_H