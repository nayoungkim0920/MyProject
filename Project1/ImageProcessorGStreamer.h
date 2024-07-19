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

class ImageProcessorGStreamer {
public:
    ImageProcessorGStreamer();
    ~ImageProcessorGStreamer();

    cv::Mat grayScale(cv::Mat& inputImage);
    cv::Mat rotate(cv::Mat& inputImage, bool isRight);
    cv::Mat zoom(cv::Mat& inputImage, double newWidth, double newHeight);
    cv::Mat cannyEdges(cv::Mat& inputImage);
    cv::Mat gaussianBlur(cv::Mat& inputImage, int kernelSize);
    cv::Mat bilateralFilter(cv::Mat& inputImage);

private:
    void drawEdgesOnColorImage(cv::Mat& image, const cv::Mat& edges);
    void drawEdgesOnGrayImage(cv::Mat& image, const cv::Mat& edges);
};

#endif // IMAGEPROCESSORGSTREAMER_H
