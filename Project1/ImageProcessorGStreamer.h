#ifndef IMAGEPROCESSORGSTREAMER_H
#define IMAGEPROCESSORGSTREAMER_H

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <opencv2/core.hpp>
#include <iostream>

class ImageProcessorGStreamer {
public:
    ImageProcessorGStreamer();
    ~ImageProcessorGStreamer();

    cv::Mat bilateralFilter(cv::Mat& inputImage);
};

#endif // IMAGEPROCESSORGSTREAMER_H
