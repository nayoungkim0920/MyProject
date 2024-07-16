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

class ImageProcessorCUDAKernel {
public:
    ImageProcessorCUDAKernel();
    ~ImageProcessorCUDAKernel();

    cv::Mat rotate(cv::Mat& inputImage);
    cv::Mat grayScale(cv::Mat& inputImage);
    cv::Mat zoom(cv::Mat& inputImage, int newWidth, int newHeight);
    cv::Mat gaussianBlur(cv::Mat& inputImage, int kernelSize);
    cv::Mat cannyEdges(cv::Mat& inputImage);
    cv::Mat medianFilter(cv::Mat& inputImage);
    cv::Mat laplacianFilter(cv::Mat& inputImage);
    cv::Mat bilateralFilter(cv::Mat& inputImage);
    cv::Mat sobelFilter(cv::Mat& inputImage);
};

#endif // IMAGEPROCESSORCUDAKERNEL_H
