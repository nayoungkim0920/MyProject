#include "ImageProcessorCUDAKernel.h"

ImageProcessorCUDAKernel::ImageProcessorCUDAKernel()
{
}

ImageProcessorCUDAKernel::~ImageProcessorCUDAKernel()
{
}

cv::Mat ImageProcessorCUDAKernel::rotate(cv::Mat& inputImage, bool isRight)
{
    std::cout << "<<<" << getClassName() << "::" << __func__ << ">>>" << std::endl;

    cv::Mat outputImage;
    if (isRight)
        callRotateImageCUDA_R(inputImage, outputImage);
    else
        callRotateImageCUDA_L(inputImage, outputImage);

    return outputImage;
}

cv::Mat ImageProcessorCUDAKernel::grayScale(cv::Mat& inputImage)
{
    std::cout << "<<<" << getClassName() << "::" << __func__ << ">>>" << std::endl;

    cv::Mat outputImage;
    callGrayScaleImageCUDA(inputImage, outputImage);

    return outputImage;
}

cv::Mat ImageProcessorCUDAKernel::zoom(cv::Mat& inputImage, int newWidth, int newHeight)
{
    std::cout << "<<<" << getClassName() << "::" << __func__ << ">>>" << std::endl;

    cv::Mat outputImage;
    callZoomImageCUDA(inputImage, outputImage, newWidth, newHeight);

    return outputImage;
}

cv::Mat ImageProcessorCUDAKernel::gaussianBlur(cv::Mat& inputImage, int kernelSize)
{
    std::cout << "<<<" << getClassName() << "::" << __func__ << ">>>" << std::endl;

    cv::Mat outputImage;
    callGaussianBlurCUDA(inputImage, outputImage, kernelSize);

    return outputImage;
}

cv::Mat ImageProcessorCUDAKernel::cannyEdges(cv::Mat& inputImage)
{
    std::cout << "<<<" << getClassName() << "::" << __func__ << ">>>" << std::endl;

    //cv::Mat grayImage;
    //if (inputImage.channels() == 3) {
    //    grayImage = grayScale(inputImage);
    //}
    //else {
    //    grayImage = inputImage.clone();
    //}

    cv::Mat outputImage;
    callCannyEdgesCUDA(inputImage, outputImage);

    return outputImage;
}

cv::Mat ImageProcessorCUDAKernel::medianFilter(cv::Mat& inputImage)
{
    std::cout << "<<<" << getClassName() << "::" << __func__ << ">>>" << std::endl;

    cv::Mat outputImage;
    callMedianFilterCUDA(inputImage, outputImage);

    return outputImage;
}

cv::Mat ImageProcessorCUDAKernel::laplacianFilter(cv::Mat& inputImage)
{
    std::cout << "<<<" << getClassName() << "::" << __func__ << ">>>" << std::endl;

    cv::Mat outputImage;
    callLaplacianFilterCUDA(inputImage, outputImage);

    // outputImage�� ����Ͽ� ���� Ȯ��
    //std::cout << "Output Image:" << std::endl;
    //std::cout << outputImage << std::endl;

    return outputImage;
    
    //ä�κи�����
    /*cv::Mat outputImage;
    // �׷��̽����� �̹����� ���
    if (inputImage.channels() == 1) {
        callLaplacianFilterCUDA(inputImage, outputImage);
    }
    // �÷� �̹����� ���
    else if (inputImage.channels() == 3) {

        std::cout << "this is color!" << std::endl;

        // ä�� �и�
        std::vector<cv::Mat> channels;
        cv::split(inputImage, channels);

        std::vector<cv::Mat> outputChannels(channels.size());
        for (int i = 0; i < channels.size(); ++i) {
            callLaplacianFilterCUDA(channels[i], outputChannels[i]);
        }

        // ä�� ����
        cv::merge(outputChannels, outputImage);
    }
    else {
        std::cerr << "Unsupported image format." << std::endl;
    }

    return outputImage;*/
}

cv::Mat ImageProcessorCUDAKernel::bilateralFilter(cv::Mat& inputImage)
{
    std::cout << "<<<" << getClassName() << "::" << __func__ << ">>>" << std::endl;

    cv::Mat outputImage;
    callBilateralFilterCUDA(inputImage, outputImage, 9, 75, 75);

    return outputImage;
}

cv::Mat ImageProcessorCUDAKernel::sobelFilter(cv::Mat& inputImage)
{
    std::cout << "<<<" << getClassName() << "::" << __func__ << ">>>" << std::endl;

    //cv::Mat grayImage = grayScale(inputImage);
    cv::Mat outputImage;
    callSobelFilterCUDA(inputImage, outputImage);

    return outputImage;
}

std::string ImageProcessorCUDAKernel::getClassName() const
{
    return "ImageProcessorCUDAKernel";
}
