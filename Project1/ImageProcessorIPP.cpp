#include "ImageProcessorIPP.h"

ImageProcessorIPP::ImageProcessorIPP()
{
}

ImageProcessorIPP::~ImageProcessorIPP()
{
}

cv::Mat ImageProcessorIPP::rotate(cv::Mat& inputImage)
{
    cv::Mat outputImage;
    cv::rotate(inputImage, outputImage, cv::ROTATE_90_CLOCKWISE);

    return outputImage;
}

cv::Mat ImageProcessorIPP::grayScale(cv::Mat& inputImage)
{
    cv::Mat outputImage;
    cv::cvtColor(inputImage, outputImage, cv::COLOR_BGR2GRAY);

    return outputImage;
}

cv::Mat ImageProcessorIPP::zoom(cv::Mat& inputImage, int newWidth, int newHeight, double x, double y, int interpolation)
{
    cv::Mat outputImage;
    cv::resize(inputImage, outputImage, cv::Size(newWidth, newHeight)
        , x, y, interpolation);

    return outputImage;
}

cv::Mat ImageProcessorIPP::gaussianBlur(cv::Mat& inputImage, int kernelSize, int sigmaX, int sigmaY, int borderType)
{
    cv::Mat outputImage;
    cv::GaussianBlur(inputImage, outputImage, cv::Size(kernelSize, kernelSize), 0, 0, 1);

    return outputImage;
}

cv::Mat ImageProcessorIPP::cannyEdges(cv::Mat& inputImage)
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

cv::Mat ImageProcessorIPP::medianFilter(cv::Mat& inputImage)
{
    cv::Mat outputImage;
    cv::medianBlur(inputImage, outputImage, 5);

    return outputImage;
}

cv::Mat ImageProcessorIPP::laplacianFilter(cv::Mat& inputImage)
{
    cv::Mat outputImage;
    cv::Laplacian(inputImage, outputImage, CV_8U, 3);

    return outputImage;
}

cv::Mat ImageProcessorIPP::bilateralFilter(cv::Mat& inputImage)
{
    cv::Mat outputImage;
    cv::bilateralFilter(inputImage, outputImage, 9, 75, 75);

    return outputImage;
}

cv::Mat ImageProcessorIPP::sobelFilter(cv::Mat& inputImage)
{
    cv::Mat gradX, gradY, absGradX, absGradY, outputImage;

    if (inputImage.channels() == 1) {
        // �׷��̽����� �̹��� ó��
        cv::Sobel(inputImage, gradX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
        cv::Sobel(inputImage, gradY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);

        cv::convertScaleAbs(gradX, absGradX);
        cv::convertScaleAbs(gradY, absGradY);

        cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, outputImage);
    }
    else {
        // �÷� �̹��� ó��
        std::vector<cv::Mat> channels, gradXChannels, gradYChannels, absGradXChannels, absGradYChannels, outputChannels;

        // �÷� ä�� �и�
        cv::split(inputImage, channels);

        // �� ä�ο� Sobel ������ ����
        for (int i = 0; i < channels.size(); ++i) {
            cv::Sobel(channels[i], gradX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
            cv::Sobel(channels[i], gradY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);

            cv::convertScaleAbs(gradX, absGradX);
            cv::convertScaleAbs(gradY, absGradY);

            gradXChannels.push_back(absGradX);
            gradYChannels.push_back(absGradY);

            // �� ä���� ����� ����
            cv::Mat outputChannel;
            cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, outputChannel);
            outputChannels.push_back(outputChannel);
        }

        // ä�� ����
        cv::merge(outputChannels, outputImage);
    }

    return outputImage;
}