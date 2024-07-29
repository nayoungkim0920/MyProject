//imageProcessing.cuh
#ifndef IMAGE_PROCESSING_CUH_
#define IMAGE_PROCESSING_CUH_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <iostream>
#include <cmath>

// CUDA 함수 호출 선언
void callRotateImageCUDA_R(cv::Mat& inputImage, cv::Mat& outputImage);
void callRotateImageCUDA_L(cv::Mat& inputImage, cv::Mat& outputImage);
void callZoomImageCUDA(cv::Mat& inputImage, cv::Mat& outputImage, int newWidth, int newHeight);
void callGrayScaleImageCUDA(cv::Mat& inputImage, cv::Mat& outputImage);
void callCannyEdgesCUDA(cv::Mat& inputImage, cv::Mat& outputImage);
void callGaussianBlurCUDA(cv::Mat& inputImage, cv::Mat& outputImage, int kernelSize);
void callMedianFilterCUDA(cv::Mat& inputImage, cv::Mat& outputImage);
void callLaplacianFilterCUDA(cv::Mat& inputImage, cv::Mat& outputImage);
void callBilateralFilterCUDA(cv::Mat& inputImage, cv::Mat& outputImage, int kernelSize, float sigmaColor, float sigmaSpace);
void callSobelFilterCUDA(cv::Mat& inputImage, cv::Mat& outputImage);


void createGaussianKernel(float* kernel, int kernelSize, float sigma);


#endif // IMAGE_PROCESSING_CUH_
