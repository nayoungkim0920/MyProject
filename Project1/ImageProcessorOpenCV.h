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

class ImageProcessorOpenCV {
public:
    ImageProcessorOpenCV();
    ~ImageProcessorOpenCV();

    cv::Mat rotate(cv::Mat& inputImage, int rotateCode);
    cv::Mat grayScale(cv::Mat& inputImage);
    cv::Mat zoom(cv::Mat& inputImage, int newWidth, int newHeight, double x, double y, int interpolation);
    cv::Mat gaussianBlur(cv::Mat& inputImage
                , int kernelSize
                , int sigmaX
                , int sigmaY
                , int borderType);
    cv::Mat cannyEdges(cv::Mat& inputImage);
    cv::Mat medianFilter(cv::Mat& inputImage);
    cv::Mat laplacianFilter(cv::Mat& inputImage); 
    cv::Mat bilateralFilter(cv::Mat& inputImage);
    cv::Mat sobelFilter(cv::Mat& inputImage);
};

#endif // IMAGEPROCESSOROPENCV_H