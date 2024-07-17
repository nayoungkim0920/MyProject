#include "ImageProcessorCUDA.h"

ImageProcessorCUDA::ImageProcessorCUDA()
{
}

ImageProcessorCUDA::~ImageProcessorCUDA()
{
}

cv::Mat ImageProcessorCUDA::rotate(cv::Mat& inputImage)
{
    // �̹����� GPU �޸𸮿� ���ε�
    cv::cuda::GpuMat gpuImage;
    gpuImage.upload(inputImage);

    // #include <opencv2/cudawarping.hpp>
    double angle = 270.0; // ȸ���� ���� (��: 90:��������90��, 270:������90��)

    // ȸ�� �߽��� �̹����� �߾����� ����
    cv::Point2f center(gpuImage.cols / 2.0f, gpuImage.rows / 2.0f);

    // ȸ�� ��Ʈ���� ���
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);

    // GPU���� ȸ�� ����
    cv::cuda::GpuMat gpuRotatedImage;
    cv::cuda::warpAffine(gpuImage, gpuRotatedImage, rotationMatrix, gpuImage.size());

    // ��� �̹����� CPU �޸𸮷� �ٿ�ε�
    cv::Mat outputImage;
    gpuRotatedImage.download(outputImage);

    return outputImage;
}

cv::Mat ImageProcessorCUDA::grayScale(cv::Mat& inputImage)
{
    cv::cuda::GpuMat d_inputImage;
    d_inputImage.upload(inputImage);

    cv::cuda::GpuMat d_outputImage;
    cv::cuda::cvtColor(d_inputImage, d_outputImage, cv::COLOR_BGR2GRAY);

    cv::Mat outputImage;
    d_outputImage.download(outputImage);

    return outputImage;
}

cv::Mat ImageProcessorCUDA::zoom(cv::Mat& inputImage, int newWidth, int newHeight)
{
    // GPU �޸𸮷� �̹��� ���ε�
    cv::cuda::GpuMat d_image;
    d_image.upload(inputImage);

    // ��� �̹����� ������ GPU �޸� �Ҵ�
    cv::cuda::GpuMat d_zoomInImage;

    // �̹��� ũ�� ����
    cv::cuda::resize(d_image, d_zoomInImage, cv::Size(static_cast<int>(newWidth), static_cast<int>(newHeight)), 0, 0, cv::INTER_LINEAR);

    // CPU �޸𸮷� ��� �̹��� �ٿ�ε�
    cv::Mat outputImage;
    d_zoomInImage.download(outputImage);

    return outputImage;
}

cv::Mat ImageProcessorCUDA::gaussianBlur(cv::Mat& inputImage, int kernelSize)
{
    cv::cuda::GpuMat gpuImage;
    gpuImage.upload(inputImage);

    cv::Ptr<cv::cuda::Filter> gaussianFilter =
        cv::cuda::createGaussianFilter(gpuImage.type()
            , gpuImage.type()
            , cv::Size(kernelSize, kernelSize)
            , 0);

    cv::cuda::GpuMat blurredGpuImage;
    gaussianFilter->apply(gpuImage, blurredGpuImage);

    cv::Mat outputImage;
    blurredGpuImage.download(outputImage);

    return outputImage;
}

cv::Mat ImageProcessorCUDA::cannyEdges(cv::Mat& inputImage)
{
    cv::Mat grayImage;
    if (inputImage.channels() == 3) {
        grayImage = grayScale(inputImage);
    }
    else {
        grayImage = inputImage.clone();
    }

    // GPU���� ĳ�� ���� ������ ����
    cv::cuda::GpuMat d_gray;
    d_gray.upload(grayImage);

    cv::cuda::GpuMat d_cannyEdges;
    cv::Ptr<cv::cuda::CannyEdgeDetector> cannyDetector = cv::cuda::createCannyEdgeDetector(50, 150);
    cannyDetector->detect(d_gray, d_cannyEdges);

    // ����� CPU �޸𸮷� ����
    cv::Mat edges;
    d_cannyEdges.download(edges);

    // ��� �̹����� �ʷϻ� ���� ǥ��
    cv::Mat outputImage = cv::Mat::zeros(inputImage.size(), CV_8UC3); // 3-channel BGR image
    cv::Mat mask(edges.size(), CV_8UC1, cv::Scalar(0)); // Mask for green edges
    mask.setTo(cv::Scalar(255), edges); // Set pixels to 255 (white) where edges are detected
    cv::Mat channels[3];
    cv::split(outputImage, channels);
    //channels[1] = mask; // Green channel is set by mask
    channels[0] = mask; // Blue channel is set by mask
    channels[1] = mask; // Green channel is set by mask
    channels[2] = mask; // Red channel is set by mask
    cv::merge(channels, 3, outputImage); // Merge channels to get green edges
    // GPU �޸� ������ GpuMat ��ü�� �������� ��� �� �ڵ����� ó���˴ϴ�.

    return outputImage;
}

cv::Mat ImageProcessorCUDA::medianFilter(cv::Mat& inputImage)
{
    // Upload image to GPU
    cv::cuda::GpuMat gpuImage;
    gpuImage.upload(inputImage);

    // Create median filter
    cv::Ptr<cv::cuda::Filter> medianFilter =
        cv::cuda::createMedianFilter(gpuImage.type(), 5);

    // Apply median filter on GPU
    cv::cuda::GpuMat medianedGpuImage;
    medianFilter->apply(gpuImage, medianedGpuImage);

    // Download the result back to CPU
    cv::Mat outputImage;
    medianedGpuImage.download(outputImage);

    return outputImage;
}

cv::Mat ImageProcessorCUDA::laplacianFilter(cv::Mat& inputImage)
{
    // �Է� �̹����� GPU �޸𸮷� ���ε�
    cv::cuda::GpuMat d_inputImage;
    d_inputImage.upload(inputImage);

    // �Է� �̹��� Ÿ�� Ȯ�� �� ä�� �� ��ȯ
    int inputType = d_inputImage.type();
    int depth = CV_MAT_DEPTH(inputType);

    // CUDA Laplacian ���͸� ������ �� �ִ� ������ Ÿ������ ��ȯ
    cv::cuda::GpuMat d_grayImage;
    if (depth != CV_8U && depth != CV_16U && depth != CV_32F) {
        d_inputImage.convertTo(d_grayImage, CV_32F);  // �Է� �̹����� CV_32F�� ��ȯ
    }
    else if (d_inputImage.channels() == 3) {
        cv::cuda::cvtColor(d_inputImage, d_grayImage, cv::COLOR_BGR2GRAY); // RGB �̹����� grayscale�� ��ȯ
    }
    else {
        d_grayImage = d_inputImage.clone();  // �̹� ������ Ÿ���� ��� �״�� ���
    }

    // Laplacian ���͸� ������ �� �Է� �� ��� �̹��� Ÿ���� �����ϰ� ����
    int srcType = d_grayImage.type();
    cv::Ptr<cv::cuda::Filter> laplacianFilter = cv::cuda::createLaplacianFilter(srcType, srcType, 3);

    // ��� �̹��� �޸� �Ҵ�
    cv::cuda::GpuMat d_outputImage(d_grayImage.size(), srcType);

    // Laplacian ���� ����
    laplacianFilter->apply(d_grayImage, d_outputImage);

    // GPU���� CPU�� ��� �̹��� �ٿ�ε�
    cv::Mat outputImage;
    d_outputImage.download(outputImage);

    return outputImage;
}

cv::Mat ImageProcessorCUDA::bilateralFilter(cv::Mat& inputImage)
{
    // �Է� �̹����� GPU �޸𸮷� ���ε�
    cv::cuda::GpuMat d_inputImage;
    d_inputImage.upload(inputImage);

    // ��� �̹��� �޸� �Ҵ�
    cv::cuda::GpuMat d_outputImage;

    // CUDA Bilateral ���� ���� �� ����
    int filterSize = 9;
    float sigmaColor = 75.0f;
    float sigmaSpace = 75.0f;

    cv::cuda::bilateralFilter(d_inputImage, d_outputImage, filterSize, sigmaColor, sigmaSpace);

    // ��� �̹����� GPU �޸𸮿��� CPU �޸𸮷� �ٿ�ε�
    cv::Mat outputImage;
    d_outputImage.download(outputImage);

    return outputImage;
}

cv::Mat ImageProcessorCUDA::sobelFilter(cv::Mat& inputImage)
{
    cv::Mat outputImage;

    // Convert input image to grayscale if necessary
    cv::Mat grayImage;
    if (inputImage.channels() > 1) {
        grayImage = grayScale(inputImage);
    }
    else {
        grayImage = inputImage;
    }

    // Transfer input image to GPU
    cv::cuda::GpuMat d_inputImage(grayImage);
    cv::cuda::GpuMat d_outputImage;

    // Create Sobel filter on GPU
    cv::Ptr<cv::cuda::Filter> sobelFilter = cv::cuda::createSobelFilter(
        d_inputImage.type(),   // srcType
        CV_16SC1,              // dstType
        1,                     // dx (order of derivative in x)
        0,                     // dy (order of derivative in y)
        3                      // ksize (kernel size, 3x3 Sobel)
    );

    // Apply Sobel filter on GPU
    sobelFilter->apply(d_inputImage, d_outputImage);

    // Transfer result back to CPU
    d_outputImage.download(outputImage);

    return outputImage;
}