#ifndef IMAGEPROCESSORGSTREAMER_H
#define IMAGEPROCESSORGSTREAMER_H

#include <gst/gst.h>
#include <gst/app/app.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <windows.h>
#include <cstdlib>

#include "AbstractImageProcessor.h"
#include "ImageProcessingLib.h"

class ImageProcessorGStreamer : public AbstractImageProcessor{
public:
    ImageProcessorGStreamer();
    ~ImageProcessorGStreamer();

    cv::Mat grayScale(cv::Mat& inputImage) override;
    cv::Mat rotate(cv::Mat& inputImage, bool isRight) override;
    cv::Mat zoom(cv::Mat& inputImage, int newWidth, int newHeight) override;
    cv::Mat cannyEdges(cv::Mat& inputImage) override;
    cv::Mat gaussianBlur(cv::Mat& inputImage, int kernelSize) override;
    cv::Mat medianFilter(cv::Mat& inputImage) override;
    cv::Mat sobelFilter(cv::Mat& inputImage) override;
    cv::Mat laplacianFilter(cv::Mat& inputImage) override;
    cv::Mat bilateralFilter(cv::Mat& inputImage) override;
private:
    std::string getClassName() const;
};

#endif // IMAGEPROCESSORGSTREAMER_H