#ifndef IMAGEPROCESSORCUDAKERNEL_H
#define IMAGEPROCESSORCUDAKERNEL_H

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>

#include "imageProcessing.cuh"
#include "AbstractImageProcessor.h"

class ImageProcessorCUDAKernel : public AbstractImageProcessor{
public:
    ImageProcessorCUDAKernel();
    ~ImageProcessorCUDAKernel();

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

#endif // IMAGEPROCESSORCUDAKERNEL_H
