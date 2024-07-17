#include "ImageProcessorCUDAKernel.h"

ImageProcessorCUDAKernel::ImageProcessorCUDAKernel()
{
}

ImageProcessorCUDAKernel::~ImageProcessorCUDAKernel()
{
}

cv::Mat ImageProcessorCUDAKernel::rotate(cv::Mat& inputImage, bool isRight)
{
    cv::Mat outputImage;
    if (isRight)
        callRotateImageCUDA_R(inputImage, outputImage);
    else
        callRotateImageCUDA_L(inputImage, outputImage);

    return outputImage;
}

cv::Mat ImageProcessorCUDAKernel::grayScale(cv::Mat& inputImage)
{
    cv::Mat outputImage;
    callGrayScaleImageCUDA(inputImage, outputImage);

    return outputImage;
}

cv::Mat ImageProcessorCUDAKernel::zoom(cv::Mat& inputImage, int newWidth, int newHeight)
{
    cv::Mat outputImage;
    callZoomImageCUDA(inputImage, outputImage, newWidth, newHeight);

    return outputImage;
}

cv::Mat ImageProcessorCUDAKernel::gaussianBlur(cv::Mat& inputImage, int kernelSize)
{
    cv::Mat outputImage;
    callGaussianBlurCUDA(inputImage, outputImage, kernelSize);

    return outputImage;
}

cv::Mat ImageProcessorCUDAKernel::cannyEdges(cv::Mat& inputImage)
{
    cv::Mat grayImage;
    if (inputImage.channels() == 3) {
        grayImage = grayScale(inputImage);
    }
    else {
        grayImage = inputImage.clone();
    }

    cv::Mat outputImage;
    callCannyEdgesCUDA(grayImage, outputImage);

    return outputImage;
}

cv::Mat ImageProcessorCUDAKernel::medianFilter(cv::Mat& inputImage)
{
    cv::Mat outputImage;
    callMedianFilterCUDA(inputImage, outputImage);

    return outputImage;
}

cv::Mat ImageProcessorCUDAKernel::laplacianFilter(cv::Mat& inputImage)
{
    cv::Mat outputImage;
    callLaplacianFilterCUDA(inputImage, outputImage);
    // outputImage를 출력하여 내용 확인
    //std::cout << "Output Image:" << std::endl;
    //std::cout << outputImage << std::endl;

    return outputImage;
}

cv::Mat ImageProcessorCUDAKernel::bilateralFilter(cv::Mat& inputImage)
{
    cv::Mat outputImage;
    callBilateralFilterCUDA(inputImage, outputImage, 9, 75, 75);

    return outputImage;
}

cv::Mat ImageProcessorCUDAKernel::sobelFilter(cv::Mat& inputImage)
{
    cv::Mat grayImage = grayScale(inputImage);
    cv::Mat outputImage;
    callSobelFilterCUDA(grayImage, outputImage);

    return outputImage;
}