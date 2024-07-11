//imageProcessing.cuh
#ifndef IMAGE_PROCESSING_CUH_
#define IMAGE_PROCESSING_CUH_

#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <iostream>

// CUDA 함수 호출 선언
void callRotateImageCUDA(cv::Mat& inputImage, cv::Mat& outputImage);
void callZoomImageCUDA(cv::Mat& inputImage, cv::Mat& outputImage, int newWidth, int newHeight);
void callGrayScaleImageCUDA(cv::Mat& inputImage, cv::Mat& outputImage);
void callCannyEdgesCUDA(cv::Mat& inputImage, cv::Mat& outputImage);
void callGaussianBlurCUDA(cv::Mat& inputImage, cv::Mat& outputImage, int kernelSize);
void callMedianFilterCUDA(cv::Mat& inputImage);
void callLaplacianFilterCUDA(cv::Mat& inputImage);
void callBilateralFilterCUDA(cv::Mat& inputImage, int kernelSize, float sigmaColor, float sigmaSpace);
void callSobelFilterCUDA(cv::Mat& inputImage);

#endif // IMAGE_PROCESSING_CUH_
