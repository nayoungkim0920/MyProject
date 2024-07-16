#include "ImageProcessorOpenCV.h"

ImageProcessorOpenCV::ImageProcessorOpenCV()
{
}

ImageProcessorOpenCV::~ImageProcessorOpenCV()
{
}

cv::Mat ImageProcessorOpenCV::rotate(cv::Mat& inputImage, int rotateCode)
{
    cv::Mat outputImage;
    cv::rotate(inputImage, outputImage, rotateCode);

    return outputImage;
}

cv::Mat ImageProcessorOpenCV::grayScale(cv::Mat& inputImage)
{
    cv::Mat outputImage;
    cv::cvtColor(inputImage, outputImage, cv::COLOR_BGR2GRAY);

    return outputImage;
}

cv::Mat ImageProcessorOpenCV::zoom(cv::Mat& inputImage, int newWidth, int newHeight, double x, double y, int interpolation)
{
    cv::Mat outputImage;
    cv::resize(inputImage, outputImage, cv::Size(newWidth, newHeight)
        , x, y, interpolation);

    return outputImage;
}

cv::Mat ImageProcessorOpenCV::gaussianBlur(cv::Mat& inputImage, int kernelSize, int sigmaX, int sigmaY, int borderType)
{
    cv::Mat outputImage;
    cv::GaussianBlur(inputImage, outputImage, cv::Size(kernelSize, kernelSize), 0, 0, 1);

    return outputImage;
}

cv::Mat ImageProcessorOpenCV::cannyEdges(cv::Mat& inputImage)
{
    cv::Mat grayImage;
    if (inputImage.channels() == 3) {
        cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
    }
    else {
        grayImage = inputImage.clone();
    }

    cv::Mat outputImage;
    cv::Canny(grayImage, outputImage, 50, 150);

    return outputImage;
}

cv::Mat ImageProcessorOpenCV::medianFilter(cv::Mat& inputImage)
{
    cv::Mat outputImage;
    cv::medianBlur(inputImage, outputImage, 5);

    return outputImage;
}

cv::Mat ImageProcessorOpenCV::laplacianFilter(cv::Mat& inputImage)
{
    cv::Mat outputImage;
    cv::Laplacian(inputImage, outputImage, CV_8U, 3);

    return outputImage;
}

cv::Mat ImageProcessorOpenCV::bilateralFilter(cv::Mat& inputImage)
{
    cv::Mat outputImage;
    cv::bilateralFilter(inputImage, outputImage, 9, 75, 75);

    return outputImage;
}

cv::Mat ImageProcessorOpenCV::sobelFilter(cv::Mat& inputImage)
{
    cv::Mat gradX, gradY, absGradX, absGradY, outputImage;

    if (inputImage.channels() == 1) {
        // 그레이스케일 이미지 처리
        cv::Sobel(inputImage, gradX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
        cv::Sobel(inputImage, gradY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);

        cv::convertScaleAbs(gradX, absGradX);
        cv::convertScaleAbs(gradY, absGradY);

        cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, outputImage);
    }
    else {
        // 컬러 이미지 처리
        std::vector<cv::Mat> channels, gradXChannels, gradYChannels, absGradXChannels, absGradYChannels, outputChannels;

        // 컬러 채널 분리
        cv::split(inputImage, channels);

        // 각 채널에 Sobel 연산자 적용
        for (int i = 0; i < channels.size(); ++i) {
            cv::Sobel(channels[i], gradX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
            cv::Sobel(channels[i], gradY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);

            cv::convertScaleAbs(gradX, absGradX);
            cv::convertScaleAbs(gradY, absGradY);

            gradXChannels.push_back(absGradX);
            gradYChannels.push_back(absGradY);

            // 각 채널의 결과를 결합
            cv::Mat outputChannel;
            cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, outputChannel);
            outputChannels.push_back(outputChannel);
        }

        // 채널 병합
        cv::merge(outputChannels, outputImage);
    }

    return outputImage;
}