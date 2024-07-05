//imageProcessing.cuh
#ifndef IMAGE_PROCESSING_CUH_
#define IMAGE_PROCESSING_CUH_

#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <iostream>

// CUDA 함수 호출 선언
extern "C" {
    void callRotateImageCUDA(cv::Mat& inputImage);
    void callResizeImageCUDA(cv::Mat& inputImage, int newWidth, int newHeight);
    void callGrayScaleImageCUDA(cv::Mat& inputImage);
    void callCannyEdgesCUDA(cv::Mat& inputImage);
    void callGaussianBlur(cv::Mat& inputImage, int kernelSize);
    void callMedianFilterCUDA(cv::Mat& inputImage);
}

// CUDA 커널 선언
__global__ void rotateImageKernel(const unsigned char* input, unsigned char* output, int cols, int rows, int channels);
__global__ void resizeImageKernel(const unsigned char* input, unsigned char* output, int oldWidth, int oldHeight, int newWidth, int newHeight, int channels);
__global__ void grayScaleImageKernel(const unsigned char* input, unsigned char* output, int cols, int rows);
__global__ void cannyEdgesKernel(const unsigned char* input, unsigned char* output, int cols, int rows);
__global__ void gaussianBlurKernel(const unsigned char* input, unsigned char* output, int cols, int rows, int kernelSize, int channels);
__global__ void medianFilterKernel(const unsigned char* input, unsigned char* output, int cols, int rows, int kernelSize, int channels);
#endif // IMAGE_PROCESSING_CUH_
