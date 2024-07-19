#ifndef IMAGEPROCESSORNPP_H
#define IMAGEPROCESSORNPP_H

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>

#include <omp.h>

#include <npp.h>
#include <nppi.h>
#include <nppi_filtering_functions.h>
#include <nppi_geometry_transforms.h>

void checkNppError(NppStatus status, const std::string& errorMessage);

class ImageProcessorNPP {
public:
    ImageProcessorNPP();
    ~ImageProcessorNPP();
    
    cv::Mat grayScale(cv::Mat& inputImage);
    cv::Mat rotate(cv::Mat& inputImage, bool isRight);
    cv::Mat zoom(cv::Mat& inputImage, double newWidth, double newHeight);
    cv::Mat gaussianBlur(cv::Mat& inputImage, int kernelSize);
    cv::Mat cannyEdges(cv::Mat& inputImage);
    cv::Mat bilateralFilter(cv::Mat& inputImage);
};

#endif // IMAGEPROCESSORIPP_H