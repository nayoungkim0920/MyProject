#include "ImageProcessorGStreamer.h"

ImageProcessorGStreamer::ImageProcessorGStreamer()
{
}

ImageProcessorGStreamer::~ImageProcessorGStreamer()
{
}

cv::Mat ImageProcessorGStreamer::grayScale(cv::Mat& inputImage) {
    GstElement* pipeline = nullptr;
    GstElement* source = nullptr;
    GstElement* convert = nullptr;
    GstElement* sink = nullptr;
    GstSample* sample = nullptr;

    gst_init(nullptr, nullptr);

    pipeline = gst_pipeline_new("pipeline");
    source = gst_element_factory_make("appsrc", "source");
    convert = gst_element_factory_make("videoconvert", "convert");
    sink = gst_element_factory_make("appsink", "sink");

    if (!pipeline || !source || !convert || !sink) {
        std::cerr << "GStreamer 요소를 생성하지 못했습니다." << std::endl;
        if (pipeline) gst_object_unref(GST_OBJECT(pipeline));
        if (source) gst_object_unref(GST_OBJECT(source));
        if (convert) gst_object_unref(GST_OBJECT(convert));
        if (sink) gst_object_unref(GST_OBJECT(sink));
        return cv::Mat();
    }

    // Set up source caps for BGR input
    GstCaps* srcCaps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "BGR",
        "width", G_TYPE_INT, inputImage.cols,
        "height", G_TYPE_INT, inputImage.rows,
        "framerate", GST_TYPE_FRACTION, 30, 1,
        nullptr);
    g_object_set(G_OBJECT(source), "caps", srcCaps, nullptr);
    g_object_set(G_OBJECT(source), "is-live", TRUE, "block", TRUE, nullptr);
    g_object_set(G_OBJECT(source), "max-bytes", inputImage.total() * inputImage.elemSize(), nullptr);

    // Set up sink caps for GRAY8 output
    GstCaps* sinkCaps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "GRAY8",
        nullptr);
    g_object_set(G_OBJECT(sink), "caps", sinkCaps, nullptr);
    g_object_set(G_OBJECT(sink), "sync", FALSE, "emit-signals", TRUE, nullptr);

    gst_bin_add_many(GST_BIN(pipeline), source, convert, sink, nullptr);
    if (!gst_element_link_many(source, convert, sink, nullptr)) {
        std::cerr << "요소들을 연결할 수 없습니다." << std::endl;
        gst_object_unref(GST_OBJECT(pipeline));
        return cv::Mat();
    }

    // Create a buffer for the input image
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, inputImage.total() * inputImage.elemSize(), nullptr);
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    std::memcpy(map.data, inputImage.data, inputImage.total() * inputImage.elemSize());
    gst_buffer_unmap(buffer, &map);

    GstFlowReturn ret;
    g_signal_emit_by_name(source, "push-buffer", buffer, &ret);
    gst_buffer_unref(buffer);

    if (ret != GST_FLOW_OK) {
        std::cerr << "appsrc에 버퍼를 푸시하는데 실패했습니다." << std::endl;
        gst_object_unref(GST_OBJECT(pipeline));
        return cv::Mat();
    }

    // Set pipeline to PLAYING state and wait for processing
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    guint64 startTime = g_get_monotonic_time();
    guint64 endTime = startTime + 5000000; // 5초 대기

    while (g_get_monotonic_time() < endTime) {
        sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
        if (sample) break;
        g_usleep(100000); // 0.1초 대기
    }

    if (!sample) {
        std::cerr << "appsink에서 샘플을 가져오는 데 실패했습니다." << std::endl;
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(GST_OBJECT(pipeline));
        return cv::Mat();
    }

    GstBuffer* outputBuffer = gst_sample_get_buffer(sample);
    if (!outputBuffer) {
        std::cerr << "샘플에서 버퍼를 가져오는 데 실패했습니다." << std::endl;
        gst_sample_unref(sample);
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(GST_OBJECT(pipeline));
        return cv::Mat();
    }

    GstMapInfo outputMap;
    gst_buffer_map(outputBuffer, &outputMap, GST_MAP_READ);

    int bufferSize = gst_buffer_get_size(outputBuffer);
    std::cout << "OutputBuffer Size: " << bufferSize << std::endl;

    // Ensure cv::Mat is created correctly
    cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC1, outputMap.data);

    // Unmap and clean up
    gst_buffer_unmap(outputBuffer, &outputMap);
    gst_sample_unref(sample);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));

    std::cout << "OutputImage Size: [" << outputImage.cols << " x " << outputImage.rows << "]" << std::endl;

    return outputImage;
}

GstBuffer* matToGstBuffer(const cv::Mat& mat) {
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, mat.total() * mat.elemSize(), nullptr);
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, mat.data, mat.total() * mat.elemSize());
    gst_buffer_unmap(buffer, &map);
    return buffer;
}

cv::Mat gstBufferToMat(GstBuffer* buffer, GstCaps* caps) {
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_READ);

    GstVideoInfo videoInfo;
    gst_video_info_from_caps(&videoInfo, caps);

    int width = GST_VIDEO_INFO_WIDTH(&videoInfo);
    int height = GST_VIDEO_INFO_HEIGHT(&videoInfo);
    int channels = GST_VIDEO_INFO_N_COMPONENTS(&videoInfo);

    cv::Mat mat(height, width, CV_8UC(channels), map.data);
    gst_buffer_unmap(buffer, &map);

    return mat;
}

cv::Mat ImageProcessorGStreamer::rotate(cv::Mat& inputImage, bool isRight) {
    GstElement* pipeline = nullptr;
    GstElement* source = nullptr;
    GstElement* convert = nullptr;
    GstElement* flip = nullptr;
    GstElement* sink = nullptr;
    GstSample* sample = nullptr;

    gst_init(nullptr, nullptr);

    pipeline = gst_pipeline_new("pipeline");
    source = gst_element_factory_make("appsrc", "source");
    convert = gst_element_factory_make("videoconvert", "convert");
    flip = gst_element_factory_make("videoflip", "flip");
    sink = gst_element_factory_make("appsink", "sink");

    if (!pipeline || !source || !convert || !flip || !sink) {
        std::cerr << "GStreamer 요소를 생성하지 못했습니다." << std::endl;
        if (pipeline) gst_object_unref(GST_OBJECT(pipeline));
        if (source) gst_object_unref(GST_OBJECT(source));
        if (convert) gst_object_unref(GST_OBJECT(convert));
        if (flip) gst_object_unref(GST_OBJECT(flip));
        if (sink) gst_object_unref(GST_OBJECT(sink));
        return cv::Mat();
    }

    // flip 속성 설정
    g_object_set(G_OBJECT(flip), "method", isRight ? 1 : 0, nullptr);

    // appsrc 속성 설정
    GstCaps* srcCaps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "BGR",
        "width", G_TYPE_INT, inputImage.cols,
        "height", G_TYPE_INT, inputImage.rows,
        "framerate", GST_TYPE_FRACTION, 30, 1,
        nullptr);
    g_object_set(G_OBJECT(source), "caps", srcCaps, nullptr);
    g_object_set(G_OBJECT(source), "is-live", TRUE, "block", TRUE, nullptr);

    // appsink 속성 설정
    GstCaps* sinkCaps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "BGR",
        nullptr);
    g_object_set(G_OBJECT(sink), "caps", sinkCaps, nullptr);
    g_object_set(G_OBJECT(sink), "sync", FALSE, "emit-signals", TRUE, nullptr);

    gst_bin_add_many(GST_BIN(pipeline), source, convert, flip, sink, nullptr);
    if (!gst_element_link_many(source, convert, flip, sink, nullptr)) {
        std::cerr << "요소들을 연결할 수 없습니다." << std::endl;
        gst_object_unref(GST_OBJECT(pipeline));
        return cv::Mat();
    }

    // cv::Mat을 GstBuffer로 변환하고 appsrc에 푸시
    GstBuffer* buffer = matToGstBuffer(inputImage);
    GstFlowReturn ret;
    g_signal_emit_by_name(source, "push-buffer", buffer, &ret);
    gst_buffer_unref(buffer);

    if (ret != GST_FLOW_OK) {
        std::cerr << "appsrc에 버퍼를 푸시하는데 실패했습니다." << std::endl;
        gst_object_unref(GST_OBJECT(pipeline));
        return cv::Mat();
    }

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    // GStreamer 파이프라인이 처리 완료될 때까지 기다림
    GstStateChangeReturn stateChangeRet;
    do {
        stateChangeRet = gst_element_get_state(pipeline, nullptr, nullptr, GST_CLOCK_TIME_NONE);
    } while (stateChangeRet != GST_STATE_CHANGE_SUCCESS);

    // appsink에서 샘플을 가져옵니다.
    sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
    if (!sample) {
        std::cerr << "appsink에서 샘플을 가져오는 데 실패했습니다." << std::endl;
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(GST_OBJECT(pipeline));
        return cv::Mat();
    }

    GstBuffer* outputBuffer = gst_sample_get_buffer(sample);
    if (!outputBuffer) {
        std::cerr << "샘플에서 버퍼를 가져오는 데 실패했습니다." << std::endl;
        gst_sample_unref(sample);
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(GST_OBJECT(pipeline));
        return cv::Mat();
    }

    GstCaps* caps = gst_sample_get_caps(sample);
    if (!caps) {
        std::cerr << "샘플에서 캡을 가져오는 데 실패했습니다." << std::endl;
        gst_sample_unref(sample);
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(GST_OBJECT(pipeline));
        return cv::Mat();
    }

    cv::Mat outputImage = gstBufferToMat(outputBuffer, caps);

    gst_sample_unref(sample);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));

    return outputImage;
}

cv::Mat ImageProcessorGStreamer::gaussianBlur(cv::Mat& inputImage, int kernelSize) {
    GstElement* pipeline = nullptr;
    GstElement* source = nullptr;
    GstElement* convert = nullptr;
    GstElement* blur = nullptr;
    GstElement* sink = nullptr;
    GstSample* sample = nullptr;

    gst_init(nullptr, nullptr);

    pipeline = gst_pipeline_new("pipeline");
    source = gst_element_factory_make("appsrc", "source");
    convert = gst_element_factory_make("videoconvert", "convert");
    blur = gst_element_factory_make("gaussianblur", "blur");  // Note: "gaussianblur" is used as a placeholder
    sink = gst_element_factory_make("appsink", "sink");

    if (!pipeline || !source || !convert || !blur || !sink) {
        std::cerr << "GStreamer 요소를 생성하지 못했습니다." << std::endl;
        if (pipeline) gst_object_unref(GST_OBJECT(pipeline));
        if (source) gst_object_unref(GST_OBJECT(source));
        if (convert) gst_object_unref(GST_OBJECT(convert));
        if (blur) gst_object_unref(GST_OBJECT(blur));
        if (sink) gst_object_unref(GST_OBJECT(sink));
        return cv::Mat();
    }

    // Gaussian blur 속성 설정 (여기서는 예시로 속성을 설정하는 코드 추가)
    g_object_set(G_OBJECT(blur), "kernel-size", kernelSize, nullptr);

    // appsrc 속성 설정
    GstCaps* srcCaps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "BGR",
        "width", G_TYPE_INT, inputImage.cols,
        "height", G_TYPE_INT, inputImage.rows,
        "framerate", GST_TYPE_FRACTION, 30, 1,
        nullptr);
    g_object_set(G_OBJECT(source), "caps", srcCaps, nullptr);
    g_object_set(G_OBJECT(source), "is-live", TRUE, "block", TRUE, nullptr);

    // appsink 속성 설정
    GstCaps* sinkCaps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "BGR",
        nullptr);
    g_object_set(G_OBJECT(sink), "caps", sinkCaps, nullptr);
    g_object_set(G_OBJECT(sink), "sync", FALSE, "emit-signals", TRUE, nullptr);

    gst_bin_add_many(GST_BIN(pipeline), source, convert, blur, sink, nullptr);
    if (!gst_element_link_many(source, convert, blur, sink, nullptr)) {
        std::cerr << "요소들을 연결할 수 없습니다." << std::endl;
        gst_object_unref(GST_OBJECT(pipeline));
        return cv::Mat();
    }

    // cv::Mat을 GstBuffer로 변환하고 appsrc에 푸시
    GstBuffer* buffer = matToGstBuffer(inputImage);
    GstFlowReturn ret;
    g_signal_emit_by_name(source, "push-buffer", buffer, &ret);
    gst_buffer_unref(buffer);

    if (ret != GST_FLOW_OK) {
        std::cerr << "appsrc에 버퍼를 푸시하는데 실패했습니다." << std::endl;
        gst_object_unref(GST_OBJECT(pipeline));
        return cv::Mat();
    }

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    // GStreamer 파이프라인이 처리 완료될 때까지 기다림
    GstStateChangeReturn stateChangeRet;
    do {
        stateChangeRet = gst_element_get_state(pipeline, nullptr, nullptr, GST_CLOCK_TIME_NONE);
    } while (stateChangeRet != GST_STATE_CHANGE_SUCCESS);

    // appsink에서 샘플을 가져옵니다.
    sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
    if (!sample) {
        std::cerr << "appsink에서 샘플을 가져오는 데 실패했습니다." << std::endl;
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(GST_OBJECT(pipeline));
        return cv::Mat();
    }

    GstBuffer* outputBuffer = gst_sample_get_buffer(sample);
    if (!outputBuffer) {
        std::cerr << "샘플에서 버퍼를 가져오는 데 실패했습니다." << std::endl;
        gst_sample_unref(sample);
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(GST_OBJECT(pipeline));
        return cv::Mat();
    }

    GstCaps* caps = gst_sample_get_caps(sample);
    if (!caps) {
        std::cerr << "샘플에서 캡을 가져오는 데 실패했습니다." << std::endl;
        gst_sample_unref(sample);
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(GST_OBJECT(pipeline));
        return cv::Mat();
    }

    cv::Mat outputImage = gstBufferToMat(outputBuffer, caps);

    gst_sample_unref(sample);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));

    return outputImage;
}

cv::Mat ImageProcessorGStreamer::bilateralFilter(cv::Mat& inputImage)
{
    GstElement* pipeline = nullptr, * source = nullptr, * convert = nullptr, * sink = nullptr;
    GstSample* sample = nullptr;

    // Initialize GStreamer
    gst_init(nullptr, nullptr);

    // Create elements
    pipeline = gst_pipeline_new("pipeline");
    source = gst_element_factory_make("appsrc", "source");
    convert = gst_element_factory_make("videoconvert", "convert");
    sink = gst_element_factory_make("appsink", "sink");

    if (!pipeline || !source || !convert || !sink) {
        std::cerr << "Failed to create elements" << std::endl;
        if (pipeline) gst_object_unref(GST_OBJECT(pipeline));
        if (source) gst_object_unref(GST_OBJECT(source));
        if (convert) gst_object_unref(GST_OBJECT(convert));
        if (sink) gst_object_unref(GST_OBJECT(sink));
        return cv::Mat(); // Return empty Mat on failure
    }

    // Set appsrc properties
    g_object_set(G_OBJECT(source), "caps",
        gst_caps_new_simple("video/x-raw",
            "format", G_TYPE_STRING, "BGR",
            "width", G_TYPE_INT, inputImage.cols,
            "height", G_TYPE_INT, inputImage.rows,
            "framerate", GST_TYPE_FRACTION, 30, 1,
            nullptr), nullptr);
    g_object_set(G_OBJECT(source), "is-live", TRUE, "block", TRUE, nullptr);

    // Set appsink properties
    g_object_set(G_OBJECT(sink), "caps",
        gst_caps_new_simple("video/x-raw",
            "format", G_TYPE_STRING, "BGR",
            nullptr), nullptr);
    g_object_set(G_OBJECT(sink), "sync", FALSE, nullptr);

    // Build the pipeline
    gst_bin_add_many(GST_BIN(pipeline), source, convert, sink, nullptr);
    if (!gst_element_link_many(source, convert, sink, nullptr)) {
        std::cerr << "Elements could not be linked." << std::endl;
        gst_object_unref(GST_OBJECT(pipeline));
        return cv::Mat();
    }

    // Apply bilateral filter
    cv::Mat filteredImage;
    cv::bilateralFilter(inputImage, filteredImage, 9, 75, 75, cv::BORDER_DEFAULT);

    // Convert cv::Mat to GstBuffer
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, filteredImage.total() * filteredImage.elemSize(), nullptr);
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, filteredImage.data, filteredImage.total() * filteredImage.elemSize());
    gst_buffer_unmap(buffer, &map);

    // Push buffer to appsrc
    GstFlowReturn ret;
    g_signal_emit_by_name(source, "push-buffer", buffer, &ret);
    gst_buffer_unref(buffer);

    if (ret != GST_FLOW_OK) {
        std::cerr << "Failed to push buffer to appsrc" << std::endl;
        gst_object_unref(GST_OBJECT(pipeline));
        return cv::Mat(); // Return empty Mat on failure
    }

    // Set the pipeline to playing
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    // Pull sample from appsink
    sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
    if (!sample) {
        std::cerr << "Failed to pull sample from appsink" << std::endl;
        gst_object_unref(GST_OBJECT(pipeline));
        return cv::Mat(); // Return empty Mat on failure
    }

    GstBuffer* outputBuffer = gst_sample_get_buffer(sample);
    if (!outputBuffer) {
        std::cerr << "Failed to get buffer from sample" << std::endl;
        gst_sample_unref(sample);
        gst_object_unref(GST_OBJECT(pipeline));
        return cv::Mat(); // Return empty Mat on failure
    }

    gst_buffer_map(outputBuffer, &map, GST_MAP_READ);
    cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC3, map.data);
    outputImage = outputImage.clone(); // Make a deep copy of the data
    gst_buffer_unmap(outputBuffer, &map);

    // Cleanup
    gst_sample_unref(sample);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    return outputImage;
}