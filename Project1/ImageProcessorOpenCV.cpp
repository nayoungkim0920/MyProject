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
    std::cout << __func__ << std::endl;

    cv::Mat grayImage;
    if (inputImage.channels() == 1)
        grayImage = inputImage.clone();
    else if(inputImage.channels() == 3)
        grayImage = grayScale(inputImage);
    else {
        std::cerr   << __func__ 
                    << " : Unsupported number of channels : " 
                    << inputImage.channels() << std::endl;

        return cv::Mat(); // 빈 이미지 반환
    }

    cv::Mat laplacianImage;
    cv::Mat outputImage;

    // Use CV_16S to prevent overflow in edge detection
    cv::Laplacian(grayImage, outputImage, CV_16S, 3);
    // Convert to CV_8U
    cv::convertScaleAbs(outputImage, outputImage);

    if (inputImage.channels() == 3) {
        // 컬러 이미지에 소벨 필터 결과를 오버레이
        cv::Mat coloredEdgeImage;
        cv::cvtColor(outputImage, coloredEdgeImage, cv::COLOR_GRAY2BGR);
        cv::addWeighted(inputImage, 0.5, coloredEdgeImage, 0.5, 0, outputImage);
    }

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
    cv::Mat grayImage;
    if (inputImage.channels() == 1)
        grayImage = inputImage.clone();
    else
        grayImage = grayScale(inputImage);

    // 소벨 필터 적용
    cv::Mat gradX, gradY, absGradX, absGradY, outputImage;

    // X 및 Y 방향의 소벨 필터 적용
    cv::Sobel(grayImage, gradX, CV_16S, 1, 0
        , 3 //kernel size
        , 1, 0, cv::BORDER_DEFAULT);
    cv::Sobel(grayImage, gradY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);

    // 절대값으로 변환
    cv::convertScaleAbs(gradX, absGradX);
    cv::convertScaleAbs(gradY, absGradY);

    // X 및 Y 방향의 그래디언트 합성
    cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, outputImage);

    if (inputImage.channels() == 3) {
        // 컬러 이미지에 소벨 필터 결과를 오버레이
        cv::Mat coloredEdgeImage;
        cv::cvtColor(outputImage, coloredEdgeImage, cv::COLOR_GRAY2BGR);
        cv::addWeighted(inputImage, 0.5, coloredEdgeImage, 0.5, 0, outputImage);
    }

    return outputImage;
}