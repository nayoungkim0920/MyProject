#include "ImageProcessorGStreamer.h"

ImageProcessorGStreamer::ImageProcessorGStreamer()
{
}

ImageProcessorGStreamer::~ImageProcessorGStreamer()
{
}

bool ImageProcessorGStreamer::initGst(GstElement*& pipeline, GstElement*& source
    , GstElement*& convert, GstElement*& sink, GstElement*& flip) {
    // GStreamer 요소 생성
    pipeline = gst_pipeline_new("pipeline");
    source = gst_element_factory_make("appsrc", "source");
    convert = gst_element_factory_make("videoconvert", "convert");
    flip = gst_element_factory_make("videoflip", "flip");
    sink = gst_element_factory_make("appsink", "sink");

    if (!pipeline || !source || !convert || !sink) {
        std::cerr << "Failed to create GStreamer elements." << std::endl;
        if (pipeline) gst_object_unref(GST_OBJECT(pipeline));
        if (source) gst_object_unref(GST_OBJECT(source));
        if (convert) gst_object_unref(GST_OBJECT(convert));
        if (flip) gst_object_unref(GST_OBJECT(flip));
        if (sink) gst_object_unref(GST_OBJECT(sink));
        return false;
    }

    return true;
}

bool ImageProcessorGStreamer::CapSet(GstElement*& source, GstCaps*& srcCaps
    , GstCaps*& sinkCaps, const cv::Mat& inputImage, GstElement*& sink
    , GstElement*& pipeline, GstElement*& convert, GstElement*& flip
    , std::string funcName) {

    if (funcName == "grayScale") {
        // BGR 입력을 위한 소스 캡 설정
        srcCaps = gst_caps_new_simple("video/x-raw",
            "format", G_TYPE_STRING, "BGR",
            "width", G_TYPE_INT, inputImage.cols,
            "height", G_TYPE_INT, inputImage.rows,
            "framerate", GST_TYPE_FRACTION, 30, 1,
            nullptr);
        g_object_set(G_OBJECT(source), "caps", srcCaps, nullptr);
        g_object_set(G_OBJECT(source), "is-live", TRUE, "block", TRUE, nullptr);

        // GRAY8 출력을 위한 싱크 캡 설정
        sinkCaps = gst_caps_new_simple("video/x-raw",
            "format", G_TYPE_STRING, "GRAY8",
            nullptr);
        g_object_set(G_OBJECT(sink), "caps", sinkCaps, nullptr);
        g_object_set(G_OBJECT(sink), "sync", FALSE, "emit-signals", TRUE, nullptr);

        gst_bin_add_many(GST_BIN(pipeline), source, convert, sink, nullptr);
        if (!gst_element_link_many(source, convert, sink, nullptr)) {
            std::cerr << "Failed to link GStreamer elements." << std::endl;
            gst_object_unref(GST_OBJECT(pipeline));
            return false;
        }
    }
    else if (funcName == "rotate" 
        || funcName == "zoom" 
        || funcName == "gaussianBlur"
        || funcName == "cannyEdges"
        || funcName == "medianFilter"
        || funcName == "sobelFilter"
        || funcName == "laplacianFilter"
        || funcName == "bilateralFilter") {

        if (inputImage.channels() == 3) {
            // 컬러 이미지인 경우
            srcCaps = gst_caps_new_simple("video/x-raw",
                "format", G_TYPE_STRING, "BGR",
                "width", G_TYPE_INT, inputImage.cols,
                "height", G_TYPE_INT, inputImage.rows,
                "framerate", GST_TYPE_FRACTION, 30, 1,
                nullptr);
            sinkCaps = gst_caps_new_simple("video/x-raw",
                "format", G_TYPE_STRING, "BGR",
                nullptr);
        }
        else if (inputImage.channels() == 1) {
            // 그레이스케일 이미지인 경우
            srcCaps = gst_caps_new_simple("video/x-raw",
                "format", G_TYPE_STRING, "GRAY8",
                "width", G_TYPE_INT, inputImage.cols,
                "height", G_TYPE_INT, inputImage.rows,
                "framerate", GST_TYPE_FRACTION, 30, 1,
                nullptr);
            sinkCaps = gst_caps_new_simple("video/x-raw",
                "format", G_TYPE_STRING, "GRAY8",
                nullptr);
        }
        g_object_set(G_OBJECT(source), "caps", srcCaps, nullptr);
        g_object_set(G_OBJECT(source), "is-live", TRUE, nullptr);
        g_object_set(G_OBJECT(sink), "caps", sinkCaps, nullptr);
        g_object_set(G_OBJECT(sink), "sync", FALSE, "emit-signals", TRUE, nullptr);

        gst_bin_add_many(GST_BIN(pipeline), source, convert, flip, sink, nullptr);
        if (!gst_element_link_many(source, convert, flip, sink, nullptr)) {
            std::cerr << "Failed to link GStreamer elements." << std::endl;
            gst_object_unref(GST_OBJECT(pipeline));
            return false;
        }
    }

    return true;
 }

bool ImageProcessorGStreamer::createBuffer(GstBuffer*& buffer
    , GstMapInfo& map, cv::Mat& inputImage, GstElement*& pipeline, GstElement*& source) {
    // 입력 이미지를 위한 버퍼 생성
    buffer = gst_buffer_new_allocate(nullptr, inputImage.total() * inputImage.elemSize(), nullptr);

    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    std::memcpy(map.data, inputImage.data, inputImage.total() * inputImage.elemSize());
    gst_buffer_unmap(buffer, &map);

    GstFlowReturn ret;
    g_signal_emit_by_name(source, "push-buffer", buffer, &ret);
    gst_buffer_unref(buffer);

    if (ret != GST_FLOW_OK) {
        std::cerr << "Failed to push buffer to appsrc." << std::endl;
        gst_object_unref(GST_OBJECT(pipeline));
        return false;
    }

    return true;
}

bool ImageProcessorGStreamer::setPipeline(GstElement*& pipeline) {
    // 파이프라인을 PLAYING 상태로 설정
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    // 처리 완료 대기
    GstStateChangeReturn stateChangeRet;
    do {
        stateChangeRet = gst_element_get_state(pipeline, nullptr, nullptr, GST_CLOCK_TIME_NONE);
        if (stateChangeRet == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "GStreamer pipeline failed!" << std::endl;
            gst_element_set_state(pipeline, GST_STATE_NULL);
            gst_object_unref(GST_OBJECT(pipeline));
            return false;
        }
    } while (stateChangeRet != GST_STATE_CHANGE_SUCCESS);
}

bool ImageProcessorGStreamer::getSample(GstElement*& sink, GstElement*& pipeline
, GstSample*& sample) {
    // appsink에서 샘플 가져오기
    sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
    if (!sample) {
        std::cerr << "Failed to pull sample from appsink." << std::endl;
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(GST_OBJECT(pipeline));
        return false;
    }
}

bool ImageProcessorGStreamer::getSampleBuffer(GstSample*& sample, GstElement*& pipeline, GstBuffer*& outputBuffer) {
    outputBuffer = gst_sample_get_buffer(sample);
    if (!outputBuffer) {
        std::cerr << "Failed to get buffer from sample." << std::endl;
        gst_sample_unref(sample);
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(GST_OBJECT(pipeline));
        return false;
    }
}

bool ImageProcessorGStreamer::sampleGetCaps(GstCaps*& caps, GstSample*& sample
    , GstElement*& pipeline) {
    caps = gst_sample_get_caps(sample);
    if (!caps) {
        std::cerr << "Failed to get caps from sample." << std::endl;
        gst_sample_unref(sample);
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(GST_OBJECT(pipeline));
        return false;
    }
}

bool ImageProcessorGStreamer::capsGetStructure(GstCaps*& caps
    , gint& width, gint& height
    , GstBuffer*& outputBuffer, GstMapInfo& outputMap
    , GstSample*& sample
    , GstElement*& pipeline){
    // 캡에서 너비와 높이 추출
    GstStructure* s = gst_caps_get_structure(caps, 0);
    if (!gst_structure_get_int(s, "width", &width) || !gst_structure_get_int(s, "height", &height)) {
        std::cerr << "Failed to extract width and height from caps." << std::endl;
        gst_buffer_unmap(outputBuffer, &outputMap);
        gst_sample_unref(sample);
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(GST_OBJECT(pipeline));
        return false;
    }
}

void ImageProcessorGStreamer::gstDestroyAll(GstBuffer*& outputBuffer
    , GstSample*& sample, GstElement*& pipeline) {

    gst_sample_unref(sample);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
}

bool ImageProcessorGStreamer::pushBufferToAppsrc(GstBuffer*& buffer, cv::Mat& inputImage
    , GstElement*& source, GstElement*& pipeline) {

    buffer = matToGstBuffer(inputImage);
    GstFlowReturn ret;
    g_signal_emit_by_name(source, "push-buffer", buffer, &ret);
    gst_buffer_unref(buffer);

    if (ret != GST_FLOW_OK) {
        std::cerr << "Failed to push buffer to appsrc." << std::endl;
        gst_object_unref(GST_OBJECT(pipeline));
        return false;
    }

    return true;
}

void ImageProcessorGStreamer::gstStatePaused(GstElement*& pipeline) {
    // 처리 완료 대기
    gst_element_set_state(pipeline, GST_STATE_PAUSED);
    GstState state;
    GstState pending;
    do {
        gst_element_get_state(pipeline, &state, &pending, GST_CLOCK_TIME_NONE);
    } while (state != GST_STATE_PAUSED && pending != GST_STATE_VOID_PENDING);
}


cv::Mat ImageProcessorGStreamer::grayScale(cv::Mat& inputImage) {

    // 입력 이미지가 이미 그레이스케일인지 확인
    if (inputImage.type() == CV_8UC1) {
        // 그레이스케일 이미지인 경우, 이미지 복제본을 반환
        std::cout << "Input image is already grayscale." << std::endl;
        return inputImage.clone(); // 이미지를 클론하여 반환
    }

    std::cout << __func__ << std::endl;
    
    gst_init(nullptr, nullptr);

    GstElement* pipeline = nullptr;
    GstElement* source = nullptr;
    GstElement* convert = nullptr;
    GstElement* sink = nullptr;
    GstSample* sample = nullptr;
    GstElement* flip = nullptr;
    
    if (!initGst(pipeline, source, convert, sink, flip))
        return cv::Mat();

    GstCaps* srcCaps;
    GstCaps* sinkCaps;

    if (!CapSet(source, srcCaps, sinkCaps, inputImage, sink
        , pipeline, convert, flip, __func__))
        return cv::Mat();    

    GstBuffer* buffer;
    GstMapInfo map;
    if (!createBuffer(buffer, map, inputImage, pipeline, source))
        return cv::Mat(); 

    if (!setPipeline(pipeline))
        return cv::Mat();   

    if (!getSample(sink, pipeline, sample))
        return cv::Mat();

    GstBuffer* outputBuffer;
    if (!getSampleBuffer(sample, pipeline, outputBuffer))
        return cv::Mat();
    
    GstCaps* caps;
    if (!sampleGetCaps(caps, sample, pipeline))
        return cv::Mat();    

    GstMapInfo outputMap;
    gst_buffer_map(outputBuffer, &outputMap, GST_MAP_READ);

    gint width;
    gint height;
    if (!capsGetStructure(caps, width, height
        , outputBuffer, outputMap
        , sample
        , pipeline))
        return cv::Mat();

    //OpenCV
    cv::Mat outputImage(height, width, CV_8UC1, outputMap.data);
    cv::Mat outputImageClone = outputImage.clone(); // 깊은 복사 생성

    gst_buffer_unmap(outputBuffer, &outputMap);
    gstDestroyAll(outputBuffer, sample, pipeline);

    return outputImageClone;
}

cv::Mat ImageProcessorGStreamer::rotate(cv::Mat& inputImage, bool isRight) {

    gst_init(nullptr, nullptr);

    GstElement* pipeline = nullptr;
    GstElement* source = nullptr;
    GstElement* convert = nullptr;
    GstElement* flip = nullptr;
    GstElement* sink = nullptr;
    GstSample* sample = nullptr;

    if (!initGst(pipeline, source, convert, sink, flip))
        return cv::Mat();

    // 회전을 위한 플립 방법 설정
    g_object_set(G_OBJECT(flip), "method", isRight ? 1 : 3, nullptr); // 1은 시계 방향 90도, 3은 반시계 방향 90도

    GstCaps* srcCaps;
    GstCaps* sinkCaps;

    if (!CapSet(source, srcCaps, sinkCaps, inputImage, sink
        , pipeline, convert, flip, __func__))
        return cv::Mat();

    GstBuffer* buffer;
    if (!pushBufferToAppsrc(buffer, inputImage, source, pipeline))
        return cv::Mat();

    if (!setPipeline(pipeline))
        return cv::Mat();

    gstStatePaused(pipeline);

    if (!getSample(sink, pipeline, sample))
        return cv::Mat();

    GstBuffer* outputBuffer;
    if (!getSampleBuffer(sample, pipeline, outputBuffer))
        return cv::Mat();

    GstCaps* caps;
    if (!sampleGetCaps(caps, sample, pipeline))
        return cv::Mat();

    cv::Mat outputImage = gstBufferToMat(outputBuffer, caps);

    gstDestroyAll(outputBuffer, sample, pipeline);

    return outputImage;
}

cv::Mat ImageProcessorGStreamer::zoom(cv::Mat& inputImage, double newWidth, double newHeight) {

    gst_init(nullptr, nullptr);

    GstElement* pipeline = nullptr;
    GstElement* source = nullptr;
    GstElement* convert = nullptr;
    GstElement* sink = nullptr;
    GstElement* flip = nullptr;
    GstSample* sample = nullptr;

    if (!initGst(pipeline, source, convert, sink, flip))
        return cv::Mat();    

    GstCaps* srcCaps;
    GstCaps* sinkCaps;

    if (!CapSet(source, srcCaps, sinkCaps, inputImage, sink
        , pipeline, convert, flip, __func__))
        return cv::Mat();

    GstBuffer* buffer;
    if (!pushBufferToAppsrc(buffer, inputImage, source, pipeline))
        return cv::Mat();   

    if (!setPipeline(pipeline))
        return cv::Mat();

    if (!getSample(sink, pipeline, sample))
        return cv::Mat();

    GstBuffer* outputBuffer;
    if (!getSampleBuffer(sample, pipeline, outputBuffer))
        return cv::Mat();

    GstCaps* caps;
    if (!sampleGetCaps(caps, sample, pipeline))
        return cv::Mat();

    cv::Mat outputImage = gstBufferToMat(outputBuffer, caps);

    // 이미지 줌 처리
    cv::Mat zoomedImage;
    cv::resize(outputImage, zoomedImage, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);

    gstDestroyAll(outputBuffer, sample, pipeline);

    return zoomedImage;
}

cv::Mat ImageProcessorGStreamer::gaussianBlur(cv::Mat& inputImage, int kernelSize) {
    
    gst_init(nullptr, nullptr);
    
    GstElement* pipeline = nullptr;
    GstElement* source = nullptr;
    GstElement* convert = nullptr;
    GstElement* sink = nullptr;
    GstElement* flip = nullptr;
    GstSample* sample = nullptr;    

    if (!initGst(pipeline, source, convert, sink, flip))
        return cv::Mat();

    GstCaps* srcCaps;
    GstCaps* sinkCaps;

    if (!CapSet(source, srcCaps, sinkCaps, inputImage, sink
        , pipeline, convert, flip, __func__))
        return cv::Mat();

    // OpenCV를 사용하여 Gaussian Blur 적용
    cv::Mat blurredImage;
    cv::GaussianBlur(inputImage, blurredImage, cv::Size(kernelSize, kernelSize), 0);

    GstBuffer* buffer;
    if (!pushBufferToAppsrc(buffer, inputImage, source, pipeline))
        return cv::Mat();
    
    if (!setPipeline(pipeline))
        return cv::Mat();

    if (!getSample(sink, pipeline, sample))
        return cv::Mat();

    GstBuffer* outputBuffer;
    if (!getSampleBuffer(sample, pipeline, outputBuffer))
        return cv::Mat();

    GstCaps* caps;
    if (!sampleGetCaps(caps, sample, pipeline))
        return cv::Mat();

    cv::Mat outputImage = gstBufferToMat(outputBuffer, caps);

    //printImagePixels(outputImage, 20);

    std::cout << "Output image type: " << outputImage.type() << std::endl;

    gstDestroyAll(outputBuffer, sample, pipeline);

    //printImagePixels(outputImage, 20);

    return outputImage;
}

cv::Mat ImageProcessorGStreamer::cannyEdges(cv::Mat& inputImage) {

    gst_init(nullptr, nullptr);

    GstElement* pipeline = nullptr;
    GstElement* source = nullptr;
    GstElement* convert = nullptr;
    GstElement* sink = nullptr;
    GstElement* flip = nullptr;
    GstSample* sample = nullptr;

    if (!initGst(pipeline, source, convert, sink, flip))
        return cv::Mat();

    GstCaps* srcCaps;
    GstCaps* sinkCaps;

    if (!CapSet(source, srcCaps, sinkCaps, inputImage, sink
        , pipeline, convert, flip, __func__))
        return cv::Mat();    

    // Apply Canny edge detection using OpenCV
    cv::Mat grayImage, outputImage;
    if (outputImage.channels() == 3) {
        cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
    }
    else {
        grayImage = inputImage.clone();
    }

    cv::Canny(grayImage, outputImage, 50, 150);

    GstBuffer* buffer;
    if (!pushBufferToAppsrc(buffer, inputImage, source, pipeline))
        return cv::Mat();

    if (!setPipeline(pipeline))
        return cv::Mat();

    if (!getSample(sink, pipeline, sample))
        return cv::Mat();

    GstBuffer* outputBuffer;
    if (!getSampleBuffer(sample, pipeline, outputBuffer))
        return cv::Mat();

    GstCaps* caps;
    if (!sampleGetCaps(caps, sample, pipeline))
        return cv::Mat();   

    cv::Mat finalOutput;
    if (inputImage.channels() == 3) {
        finalOutput = inputImage.clone();
        for (int y = 0; y < outputImage.rows; ++y) {
            for (int x = 0; x < outputImage.cols; ++x) {
                if (outputImage.at<uchar>(y, x) > 0) {
                    finalOutput.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0); // Green
                }
            }
        }
    }
    else {
        finalOutput = cv::Mat(grayImage.size(), CV_8UC1, cv::Scalar(0));
        for (int y = 0; y < outputImage.rows; ++y) {
            for (int x = 0; x < outputImage.cols; ++x) {
                if (outputImage.at<uchar>(y, x) > 0) {
                    finalOutput.at<uchar>(y, x) = 255; // White
                }
            }
        }
    }

    gstDestroyAll(outputBuffer, sample, pipeline);

    return finalOutput;
}

cv::Mat ImageProcessorGStreamer::medianFilter(cv::Mat& inputImage) {

    gst_init(nullptr, nullptr);

    GstElement* pipeline = nullptr;
    GstElement* source = nullptr;
    GstElement* convert = nullptr;
    GstElement* sink = nullptr;
    GstElement* flip = nullptr;
    GstSample* sample = nullptr;

    if (!initGst(pipeline, source, convert, sink, flip))
        return cv::Mat();

    GstCaps* srcCaps;
    GstCaps* sinkCaps;

    if (!CapSet(source, srcCaps, sinkCaps, inputImage, sink
        , pipeline, convert, flip, __func__))
        return cv::Mat();

    cv::Mat outputImage = inputImage.clone();
    cv::medianBlur(inputImage, outputImage, 5);

    GstBuffer* buffer;
    if (!pushBufferToAppsrc(buffer, inputImage, source, pipeline))
        return cv::Mat();

    if (!setPipeline(pipeline))
        return cv::Mat();

    if (!getSample(sink, pipeline, sample))
        return cv::Mat();

    GstBuffer* outputBuffer;
    if (!getSampleBuffer(sample, pipeline, outputBuffer))
        return cv::Mat();

    GstCaps* caps;
    if (!sampleGetCaps(caps, sample, pipeline))
        return cv::Mat();

    std::cout << "Output image type: " << outputImage.type() << std::endl;

    gstDestroyAll(outputBuffer, sample, pipeline);

    return outputImage;    
}

cv::Mat ImageProcessorGStreamer::sobelFilter(cv::Mat& inputImage)
{
    gst_init(nullptr, nullptr);

    GstElement* pipeline = nullptr;
    GstElement* source = nullptr;
    GstElement* convert = nullptr;
    GstElement* sink = nullptr;
    GstElement* flip = nullptr;
    GstSample* sample = nullptr;

    if (!initGst(pipeline, source, convert, sink, flip))
        return cv::Mat();

    GstCaps* srcCaps;
    GstCaps* sinkCaps;

    if (!CapSet(source, srcCaps, sinkCaps, inputImage, sink
        , pipeline, convert, flip, __func__))
        return cv::Mat();    

    // Apply Sobel filter using OpenCV
    cv::Mat gradX, gradY, absGradX, absGradY, outputImage;

    // Apply Sobel filter
    cv::Sobel(inputImage, gradX, CV_16S, 1, 0, 3);
    cv::convertScaleAbs(gradX, absGradX);
    cv::Sobel(inputImage, gradY, CV_16S, 0, 1, 3);
    cv::convertScaleAbs(gradY, absGradY);
    cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, outputImage);

    GstBuffer* buffer;
    GstMapInfo map;
    if (!createBuffer(buffer, map, outputImage, pipeline, source))
       return cv::Mat();   

    // Set the pipeline to playing
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    if (!getSample(sink, pipeline, sample))
        return cv::Mat();

    GstBuffer* outputBuffer;
    if (!getSampleBuffer(sample, pipeline, outputBuffer))
        return cv::Mat();

    //gst_buffer_map(outputBuffer, &map, GST_MAP_READ);
    //cv::Mat finalImage(inputImage.rows, inputImage.cols, CV_8UC3, map.data);
    //finalImage = finalImage.clone(); // Make a deep copy of the data
    //gst_buffer_unmap(outputBuffer, &map);

    gstDestroyAll(outputBuffer, sample, pipeline);

    return outputImage;

}

cv::Mat ImageProcessorGStreamer::laplacianFilter(cv::Mat& inputImage)
{    
    gst_init(nullptr, nullptr);

    GstElement* pipeline = nullptr;
    GstElement* source = nullptr;
    GstElement* convert = nullptr;
    GstElement* sink = nullptr;
    GstElement* flip = nullptr;
    GstSample* sample = nullptr;

    if (!initGst(pipeline, source, convert, sink, flip))
        return cv::Mat();

    GstCaps* srcCaps;
    GstCaps* sinkCaps;

    if (!CapSet(source, srcCaps, sinkCaps, inputImage, sink
        , pipeline, convert, flip, __func__))
        return cv::Mat();

    cv::Mat outputImage;

    if (inputImage.channels() == 3) {
        // Apply Laplacian filter to each channel of the color image
        std::vector<cv::Mat> channels(3);
        cv::split(inputImage, channels);

        std::vector<cv::Mat> laplacianChannels(3);
        for (int i = 0; i < 3; ++i) {
            cv::Mat grayChannel;
            cv::Laplacian(channels[i], grayChannel, CV_16S, 3);
            cv::convertScaleAbs(grayChannel, laplacianChannels[i]);
        }

        cv::merge(laplacianChannels, outputImage);
    }
    else {
        // For grayscale image
        cv::Laplacian(inputImage, outputImage, CV_16S, 3);
        cv::convertScaleAbs(outputImage, outputImage);
    }

    // Convert cv::Mat to GstBuffer
    GstBuffer* buffer;
    GstMapInfo map;
    if (!createBuffer(buffer, map, inputImage, pipeline, source))
        return cv::Mat();

    // Set the pipeline to PLAYING state
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    // Pull sample from appsink
    if (!getSample(sink, pipeline, sample))
        return cv::Mat();

    GstBuffer* outputBuffer;
    if (!getSampleBuffer(sample, pipeline, outputBuffer))
        return cv::Mat();

    //gst_buffer_map(outputBuffer, &map, GST_MAP_READ);
    //cv::Mat finalImage(inputImage.rows, inputImage.cols, CV_8UC1, map.data);
    //finalImage = finalImage.clone(); // Make a deep copy of the data
    //gst_buffer_unmap(outputBuffer, &map);

    // Cleanup
    gstDestroyAll(outputBuffer, sample, pipeline);

    return outputImage;
}

cv::Mat ImageProcessorGStreamer::bilateralFilter(cv::Mat& inputImage)
{
    gst_init(nullptr, nullptr);

    GstElement* pipeline = nullptr;
    GstElement* source = nullptr;
    GstElement* convert = nullptr;
    GstElement* sink = nullptr;
    GstElement* flip = nullptr;
    GstSample* sample = nullptr;

    if (!initGst(pipeline, source, convert, sink, flip))
        return cv::Mat();

    GstCaps* srcCaps;
    GstCaps* sinkCaps;

    if (!CapSet(source, srcCaps, sinkCaps, inputImage, sink
        , pipeline, convert, flip, __func__))
        return cv::Mat();

    // 양방향 필터 적용
    cv::Mat outputImage;
    cv::bilateralFilter(inputImage, outputImage, 9, 75, 75, cv::BORDER_DEFAULT);

    // cv::Mat을 GstBuffer로 변환
    GstBuffer* buffer;
    GstMapInfo map;
    if (!createBuffer(buffer, map, inputImage, pipeline, source))
        return cv::Mat();

    // 파이프라인을 재생 상태로 설정
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    if (!getSample(sink, pipeline, sample))
        return cv::Mat();

    GstBuffer* outputBuffer;
    if (!getSampleBuffer(sample, pipeline, outputBuffer))
        return cv::Mat();

    //gst_buffer_map(outputBuffer, &map, GST_MAP_READ);
    //cv::Mat outputImage(inputImage.rows, inputImage.cols, inputImage.type(), map.data);
    //outputImage = outputImage.clone(); // 데이터의 깊은 복사본 만들기
    //gst_buffer_unmap(outputBuffer, &map);

    // 정리
    gstDestroyAll(outputBuffer, sample, pipeline);

    return outputImage;
}