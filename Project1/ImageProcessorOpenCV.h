#ifndef IMAGEPROCESSOROPENCV_H
#define IMAGEPROCESSOROPENCV_H

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>

#include "AbstractImageProcessor.h"

class ImageProcessorOpenCV : public AbstractImageProcessor {
public:
    ImageProcessorOpenCV();
    ~ImageProcessorOpenCV();

    // 추상 클래스에서 정의한 함수들
    cv::Mat rotate(cv::Mat& inputImage, bool isRight) override;
    cv::Mat grayScale(cv::Mat& inputImage) override;
    cv::Mat zoom(cv::Mat& inputImage, int newWidth, int newHeight) override;
    cv::Mat gaussianBlur(cv::Mat& inputImage, int kernelSize) override;
    cv::Mat cannyEdges(cv::Mat& inputImage) override;
    cv::Mat medianFilter(cv::Mat& inputImage) override;
    cv::Mat laplacianFilter(cv::Mat& inputImage) override;
    cv::Mat bilateralFilter(cv::Mat& inputImage) override;
    cv::Mat sobelFilter(cv::Mat& inputImage) override;

    // 오버로딩된 메소드들
    cv::Mat zoom(cv::Mat& inputImage, int newWidth, int newHeight
        , double x, double y, int interpolation);
    cv::Mat gaussianBlur(cv::Mat& inputImage, int kernelSize
        , int sigmaX, int sigmaY, int borderType);

private:
    std::string getClassName() const override;
};

#endif // IMAGEPROCESSOROPENCV_H
