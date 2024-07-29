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

#include "ImageProcessingLib.h"

class ImageProcessorGStreamer {
public:
    ImageProcessorGStreamer();
    ~ImageProcessorGStreamer();

    cv::Mat grayScale(cv::Mat& inputImage);
    cv::Mat rotate(cv::Mat& inputImage, bool isRight);
    cv::Mat zoom(cv::Mat& inputImage, double newWidth, double newHeight);
    cv::Mat cannyEdges(cv::Mat& inputImage);
    cv::Mat gaussianBlur(cv::Mat& inputImage, int kernelSize);
    cv::Mat medianFilter(cv::Mat& inputImage);
    cv::Mat sobelFilter(cv::Mat& inputImage);
    cv::Mat laplacianFilter(cv::Mat& inputImage);
    cv::Mat bilateralFilter(cv::Mat& inputImage);

private:
    bool initGst(GstElement*& pipeline, GstElement*& source
        , GstElement*& convert, GstElement*& sink, GstElement*& flip);
    bool CapSet(GstElement*& source, GstCaps*& srcCaps
        , GstCaps*& sinkCaps, const cv::Mat& inputImage, GstElement*& sink
        , GstElement*& pipeline, GstElement*& convert, GstElement*& flip
        , std::string funcName);
    bool createBuffer(GstBuffer*& buffer, GstMapInfo& map, cv::Mat& inputImage
        , GstElement*& pipeline, GstElement*& source);
    bool setPipeline(GstElement*& pipeline);
    bool getSample(GstElement*& sink, GstElement*& pipeline, GstSample*& sample);
    bool getSampleBuffer(GstSample*& sample, GstElement*& pipeline, GstBuffer*& outputBuffer);
    bool sampleGetCaps(GstCaps*& caps, GstSample*& sample, GstElement*& pipeline);
    bool capsGetStructure(GstCaps*& caps, gint& width, gint& height
        , GstBuffer*& outputBuffer, GstMapInfo& outputMap
        , GstSample*& sample
        , GstElement*& pipeline);
    void gstDestroyAll(GstBuffer*& outputBuffer 
        , GstSample*& sample, GstElement*& pipeline);
    bool pushBufferToAppsrc(GstBuffer*& buffer, cv::Mat& inputImage
        , GstElement*& source, GstElement*& pipeline);
    void gstStatePaused(GstElement*& pipeline);
};

#endif // IMAGEPROCESSORGSTREAMER_H
