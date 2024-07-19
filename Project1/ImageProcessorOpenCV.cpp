#include "ImageProcessorOpenCV.h"

ImageProcessorOpenCV::ImageProcessorOpenCV()
{
}

ImageProcessorOpenCV::~ImageProcessorOpenCV()
{
}

cv::Mat ImageProcessorOpenCV::rotate(cv::Mat& inputImage, bool isRight)
{
    int rotateCode;

    if (isRight)
        rotateCode = cv::ROTATE_90_CLOCKWISE;
    else
        rotateCode = cv::ROTATE_90_COUNTERCLOCKWISE;

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
    cv::Mat edges;
    cv::Mat outputImage = inputImage.clone();

    if (inputImage.channels() == 3) {
        // 컬러 이미지인 경우, 그레이스케일로 변환하여 Canny 엣지 검출 수행
        cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
        cv::Canny(grayImage, edges, 50, 150);

        // 엣지를 초록색으로 표시
        for (int y = 0; y < edges.rows; y++) {
            for (int x = 0; x < edges.cols; x++) {
                if (edges.at<uchar>(y, x) > 0) {
                    outputImage.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0); // 초록색
                }
            }
        }
    }
    else if (inputImage.channels() == 1) {
        // 회색조 이미지인 경우
        cv::Canny(inputImage, outputImage, 50, 150);
    }
    else {
        throw std::runtime_error("지원되지 않는 이미지 형식입니다.");
    }

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