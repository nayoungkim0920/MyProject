#include "ImageProcessorCUDA.h"

ImageProcessorCUDA::ImageProcessorCUDA()
{
}

ImageProcessorCUDA::~ImageProcessorCUDA()
{
}

cv::Mat ImageProcessorCUDA::rotate(cv::Mat& inputImage)
{
    // 이미지를 GPU 메모리에 업로드
    cv::cuda::GpuMat gpuImage;
    gpuImage.upload(inputImage);

    // #include <opencv2/cudawarping.hpp>
    double angle = 270.0; // 회전할 각도 (예: 90:왼쪽으로90도, 270:오른쪽90도)

    // 회전 중심을 이미지의 중앙으로 설정
    cv::Point2f center(gpuImage.cols / 2.0f, gpuImage.rows / 2.0f);

    // 회전 매트릭스 계산
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);

    // GPU에서 회전 수행
    cv::cuda::GpuMat gpuRotatedImage;
    cv::cuda::warpAffine(gpuImage, gpuRotatedImage, rotationMatrix, gpuImage.size());

    // 결과 이미지를 CPU 메모리로 다운로드
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
    // GPU 메모리로 이미지 업로드
    cv::cuda::GpuMat d_image;
    d_image.upload(inputImage);

    // 결과 이미지를 저장할 GPU 메모리 할당
    cv::cuda::GpuMat d_zoomInImage;

    // 이미지 크기 조정
    cv::cuda::resize(d_image, d_zoomInImage, cv::Size(static_cast<int>(newWidth), static_cast<int>(newHeight)), 0, 0, cv::INTER_LINEAR);

    // CPU 메모리로 결과 이미지 다운로드
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

    // GPU에서 캐니 엣지 감지기 생성
    cv::cuda::GpuMat d_gray;
    d_gray.upload(grayImage);

    cv::cuda::GpuMat d_cannyEdges;
    cv::Ptr<cv::cuda::CannyEdgeDetector> cannyDetector = cv::cuda::createCannyEdgeDetector(50, 150);
    cannyDetector->detect(d_gray, d_cannyEdges);

    // 결과를 CPU 메모리로 복사
    cv::Mat edges;
    d_cannyEdges.download(edges);

    // 출력 이미지에 초록색 엣지 표시
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
    // GPU 메모리 해제는 GpuMat 객체가 스코프를 벗어날 때 자동으로 처리됩니다.

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
    // 입력 이미지를 GPU 메모리로 업로드
    cv::cuda::GpuMat d_inputImage;
    d_inputImage.upload(inputImage);

    // 입력 이미지 타입 확인 및 채널 수 변환
    int inputType = d_inputImage.type();
    int depth = CV_MAT_DEPTH(inputType);

    // CUDA Laplacian 필터를 적용할 수 있는 데이터 타입으로 변환
    cv::cuda::GpuMat d_grayImage;
    if (depth != CV_8U && depth != CV_16U && depth != CV_32F) {
        d_inputImage.convertTo(d_grayImage, CV_32F);  // 입력 이미지를 CV_32F로 변환
    }
    else if (d_inputImage.channels() == 3) {
        cv::cuda::cvtColor(d_inputImage, d_grayImage, cv::COLOR_BGR2GRAY); // RGB 이미지를 grayscale로 변환
    }
    else {
        d_grayImage = d_inputImage.clone();  // 이미 적절한 타입인 경우 그대로 사용
    }

    // Laplacian 필터를 생성할 때 입력 및 출력 이미지 타입을 동일하게 설정
    int srcType = d_grayImage.type();
    cv::Ptr<cv::cuda::Filter> laplacianFilter = cv::cuda::createLaplacianFilter(srcType, srcType, 3);

    // 출력 이미지 메모리 할당
    cv::cuda::GpuMat d_outputImage(d_grayImage.size(), srcType);

    // Laplacian 필터 적용
    laplacianFilter->apply(d_grayImage, d_outputImage);

    // GPU에서 CPU로 결과 이미지 다운로드
    cv::Mat outputImage;
    d_outputImage.download(outputImage);

    return outputImage;
}

cv::Mat ImageProcessorCUDA::bilateralFilter(cv::Mat& inputImage)
{
    // 입력 이미지를 GPU 메모리로 업로드
    cv::cuda::GpuMat d_inputImage;
    d_inputImage.upload(inputImage);

    // 출력 이미지 메모리 할당
    cv::cuda::GpuMat d_outputImage;

    // CUDA Bilateral 필터 설정 및 적용
    int filterSize = 9;
    float sigmaColor = 75.0f;
    float sigmaSpace = 75.0f;

    cv::cuda::bilateralFilter(d_inputImage, d_outputImage, filterSize, sigmaColor, sigmaSpace);

    // 출력 이미지를 GPU 메모리에서 CPU 메모리로 다운로드
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