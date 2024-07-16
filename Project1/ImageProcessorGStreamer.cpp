#include "ImageProcessorGStreamer.h"

ImageProcessorGStreamer::ImageProcessorGStreamer()
{
}

ImageProcessorGStreamer::~ImageProcessorGStreamer()
{
}

cv::Mat ImageProcessorGStreamer::bilateralFilter(cv::Mat& inputImage)
{
    // Initialize GStreamer pipeline elements
    GstElement* pipeline = nullptr, * source = nullptr, * filter = nullptr, * sink = nullptr;
    GstBus* bus = nullptr;
    GstMessage* msg = nullptr;
    gboolean terminate = FALSE;

    // Initialize GStreamer
    gst_init(nullptr, nullptr);

    // Create elements
    pipeline = gst_pipeline_new("pipeline");
    source = gst_element_factory_make("appsrc", "source");
    filter = gst_element_factory_make("videoconvert", "filter");
    sink = gst_element_factory_make("appsink", "sink");

    if (!pipeline || !source || !filter || !sink) {
        std::cerr << "Failed to create elements" << std::endl;
        if (pipeline) gst_object_unref(GST_OBJECT(pipeline));
        if (source) gst_object_unref(GST_OBJECT(source));
        if (filter) gst_object_unref(GST_OBJECT(filter));
        if (sink) gst_object_unref(GST_OBJECT(sink));
        return cv::Mat(); // Return empty Mat on failure
    }

    // Build the pipeline
    gst_bin_add_many(GST_BIN(pipeline), source, filter, sink, nullptr);
    if (gst_element_link_many(source, filter, sink, nullptr) != TRUE) {
        std::cerr << "Failed to link elements" << std::endl;
        gst_object_unref(GST_OBJECT(pipeline));
        return cv::Mat(); // Return empty Mat on failure
    }

    // Set appsrc properties
    g_object_set(G_OBJECT(source), "format", GST_FORMAT_TIME, nullptr);

    // Prepare AppSrc
    GstCaps* caps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "BGR",
        "width", G_TYPE_INT, inputImage.cols,
        "height", G_TYPE_INT, inputImage.rows,
        "framerate", GST_TYPE_FRACTION, 0, 1,
        nullptr);
    g_object_set(G_OBJECT(source), "caps", caps, nullptr);
    gst_caps_unref(caps);

    // Convert cv::Mat to GstBuffer
    GstBuffer* buffer;
    buffer = gst_buffer_new_wrapped_full(GST_MEMORY_FLAG_READONLY,
        inputImage.data,
        inputImage.cols * inputImage.rows * inputImage.elemSize(),
        0,
        inputImage.cols * inputImage.rows * inputImage.elemSize(),
        nullptr,
        nullptr);

    // Push buffer to AppSrc
    GstFlowReturn ret;
    g_signal_emit_by_name(source, "push-buffer", buffer, &ret);
    gst_buffer_unref(buffer);

    // Get output from AppSink
    GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
    GstBuffer* outputBuffer = gst_sample_get_buffer(sample);
    GstMapInfo map;
    gst_buffer_map(outputBuffer, &map, GST_MAP_READ);

    // Convert GstBuffer to cv::Mat
    cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC3, map.data);

    // Cleanup
    gst_buffer_unmap(outputBuffer, &map);
    gst_sample_unref(sample);
    gst_object_unref(bus);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    return outputImage;
}