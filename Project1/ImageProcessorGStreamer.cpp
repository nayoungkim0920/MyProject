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

    // Wait for the processing to complete
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

    // Set flip method for rotation
    g_object_set(G_OBJECT(flip), "method", isRight ? 1 : 0, nullptr); // 1 for 90 degrees clockwise, 0 for 90 degrees counter-clockwise

    // Set appsrc properties
    GstCaps* srcCaps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "BGR",
        "width", G_TYPE_INT, inputImage.cols,
        "height", G_TYPE_INT, inputImage.rows,
        "framerate", GST_TYPE_FRACTION, 30, 1,
        nullptr);
    g_object_set(G_OBJECT(source), "caps", srcCaps, nullptr);
    g_object_set(G_OBJECT(source), "is-live", TRUE, nullptr);

    // Set appsink properties
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

    // Convert cv::Mat to GstBuffer and push to appsrc
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

    // Poll for the pipeline's state change
    GstStateChangeReturn stateChangeRet;
    do {
        
        stateChangeRet = gst_element_get_state(pipeline, nullptr, nullptr, GST_CLOCK_TIME_NONE);
        if (stateChangeRet == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "GStreamer pipeline failed!" << std::endl;
            gst_element_set_state(pipeline, GST_STATE_NULL);
            gst_object_unref(GST_OBJECT(pipeline));
            return cv::Mat();
        }
    } while (stateChangeRet != GST_STATE_CHANGE_SUCCESS);
    std::cerr << "*************************" << std::endl;

    // Wait for processing to complete
    gst_element_set_state(pipeline, GST_STATE_PAUSED);
    GstState state;
    GstState pending;
    do {
        gst_element_get_state(pipeline, &state, &pending, GST_CLOCK_TIME_NONE);
    } while (state != GST_STATE_PAUSED && pending != GST_STATE_VOID_PENDING);

    // Pull sample from appsink
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

cv::Mat ImageProcessorGStreamer::zoom(cv::Mat& inputImage, double newWidth, double newHeight) {

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

    // appsrc 설정
    GstCaps* srcCaps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "BGR",
        "width", G_TYPE_INT, inputImage.cols,
        "height", G_TYPE_INT, inputImage.rows,
        "framerate", GST_TYPE_FRACTION, 30, 1,
        nullptr);
    g_object_set(G_OBJECT(source), "caps", srcCaps, nullptr);
    g_object_set(G_OBJECT(source), "is-live", TRUE, "block", TRUE, nullptr);

    // appsink 설정
    GstCaps* sinkCaps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "BGR",
        nullptr);
    g_object_set(G_OBJECT(sink), "caps", sinkCaps, nullptr);
    g_object_set(G_OBJECT(sink), "sync", FALSE, "emit-signals", TRUE, nullptr);

    gst_bin_add_many(GST_BIN(pipeline), source, convert, sink, nullptr);
    if (!gst_element_link_many(source, convert, sink, nullptr)) {
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
        if (stateChangeRet == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "GStreamer 파이프라인 상태 변경 실패." << std::endl;
            gst_element_set_state(pipeline, GST_STATE_NULL);
            gst_object_unref(GST_OBJECT(pipeline));
            return cv::Mat();
        }
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

    // 이미지 줌 처리
    cv::Mat zoomedImage;
    cv::resize(outputImage, zoomedImage, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);

    return zoomedImage;
}

cv::Mat ImageProcessorGStreamer::gaussianBlur(cv::Mat& inputImage, int kernelSize) {
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
        std::cerr << "GStreamer failed" << std::endl;
        return cv::Mat();
    }

    std::cout << "GStreamer 파이프라인 요소 생성 성공." << std::endl;

    // 입력 이미지의 캡 설정
    GstCaps* srcCaps = nullptr;
    switch (inputImage.type()) {
    case CV_8UC1:
        srcCaps = gst_caps_new_simple("video/x-raw",
            "format", G_TYPE_STRING, "GRAY8",
            "width", G_TYPE_INT, inputImage.cols,
            "height", G_TYPE_INT, inputImage.rows,
            "framerate", GST_TYPE_FRACTION, 30, 1,
            nullptr);
        break;
    case CV_8UC3:
        srcCaps = gst_caps_new_simple("video/x-raw",
            "format", G_TYPE_STRING, "BGR",
            "width", G_TYPE_INT, inputImage.cols,
            "height", G_TYPE_INT, inputImage.rows,
            "framerate", GST_TYPE_FRACTION, 30, 1,
            nullptr);
        break;
    default:
        std::cerr << "지원하지 않는 이미지 타입입니다: " << inputImage.type() << std::endl;
        gst_object_unref(GST_OBJECT(pipeline));
        return cv::Mat();
    }

    std::cout << "Source caps set: " << gst_caps_to_string(srcCaps) << std::endl;
    g_object_set(G_OBJECT(source), "caps", srcCaps, nullptr);
    g_object_set(G_OBJECT(source), "is-live", TRUE, nullptr);

    GstCaps* sinkCaps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, inputImage.type() == CV_8UC1 ? "GRAY8" : "BGR",
        nullptr);
    std::cout << "Sink caps set: " << gst_caps_to_string(sinkCaps) << std::endl;
    g_object_set(G_OBJECT(sink), "caps", sinkCaps, nullptr);
    g_object_set(G_OBJECT(sink), "sync", TRUE, "emit-signals", TRUE, nullptr);

    gst_bin_add_many(GST_BIN(pipeline), source, convert, sink, nullptr);
    if (!gst_element_link_many(source, convert, sink, nullptr)) {
        std::cerr << "요소들을 연결할 수 없습니다." << std::endl;
        gst_object_unref(GST_OBJECT(pipeline));
        return cv::Mat();
    }

    // OpenCV를 사용하여 Gaussian Blur 적용
    cv::Mat blurredImage;
    cv::GaussianBlur(inputImage, blurredImage, cv::Size(kernelSize, kernelSize), 0);

    // 처리된 이미지를 GStreamer 파이프라인으로 푸시
    GstBuffer* buffer = matToGstBuffer(blurredImage);
    GstFlowReturn ret;
    g_signal_emit_by_name(source, "push-buffer", buffer, &ret);
    gst_buffer_unref(buffer);

    if (ret != GST_FLOW_OK) {
        std::cerr << "appsrc에 버퍼를 푸시하는데 실패했습니다. 반환값: " << ret << std::endl;
        gst_object_unref(GST_OBJECT(pipeline));
        return cv::Mat();
    }

    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    std::cout << "Pipeline 상태를 PLAYING으로 설정했습니다." << std::endl;

    GstStateChangeReturn stateChangeRet;
    do {
        stateChangeRet = gst_element_get_state(pipeline, nullptr, nullptr, GST_CLOCK_TIME_NONE);
        std::cout << "Pipeline 상태 변경: " << stateChangeRet << std::endl;
    } while (stateChangeRet != GST_STATE_CHANGE_SUCCESS);

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

    printImagePixels(outputImage, 20);

    std::cout << "Output image type: " << outputImage.type() << std::endl;

    gst_sample_unref(sample);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));

    printImagePixels(outputImage, 20);

    return outputImage;
}

cv::Mat ImageProcessorGStreamer::cannyEdges(cv::Mat& inputImage) {

    cv::Mat grayImage;
    cv::Mat edges;
    cv::Mat outputImage;

    // 입력 이미지가 컬러일 경우 그레이스케일로 변환
    if (inputImage.channels() == 3) {
        cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
    }
    else {
        grayImage = inputImage.clone();
    }

    // Canny 엣지 검출 수행
    cv::Canny(grayImage, edges, 50, 150);

    // Canny 엣지 결과를 출력 이미지로 변환
    if (inputImage.channels() == 3) {
        // 컬러 이미지에 초록색 엣지 표시
        outputImage = inputImage.clone();
        drawEdgesOnColorImage(outputImage, edges);
    }
    else {
        // 그레이스케일 이미지에 흰색 엣지 표시
        outputImage.create(edges.size(), CV_8UC1);
        drawEdgesOnGrayImage(outputImage, edges);
    }

    return outputImage;
}

cv::Mat ImageProcessorGStreamer::medianFilter(cv::Mat& inputImage) {

    cv::Mat outputImage;

    if (inputImage.empty()) {
        std::cerr << "Input image is empty." << std::endl;
        return outputImage; // 빈 이미지 반환
    }

    int numChannels = inputImage.channels();
    int kernelSize = 5; // 미디안 필터 커널 크기

    // 그레이스케일 이미지 처리
    if (numChannels == 1) {
        cv::medianBlur(inputImage, outputImage, kernelSize);
    }
    // 컬러 이미지 처리
    else if (numChannels == 3) {
        std::vector<cv::Mat> channels(3);
        cv::split(inputImage, channels);

        for (int i = 0; i < 3; ++i) {
            cv::medianBlur(channels[i], channels[i], kernelSize);
        }

        cv::merge(channels, outputImage);
    }
    else {
        std::cerr << "Unsupported number of channels: " << numChannels << std::endl;
    }

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