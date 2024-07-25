#include "ImageProcessorNPP.h"

ImageProcessorNPP::ImageProcessorNPP()
{
}

ImageProcessorNPP::~ImageProcessorNPP()
{
}

cv::Mat ImageProcessorNPP::grayScale(cv::Mat& inputImage) {
    if (inputImage.channels() != 3) {
        std::cerr << "입력 이미지는 컬러 이미지여야 합니다." << std::endl;
        return cv::Mat();
    }

    // CUDA 스트림 생성
    cudaStream_t cudaStream;

    cudaStreamCreate(&cudaStream);

    // CUDA 메모리 할당
    Npp8u* d_src = nullptr;
    Npp8u* d_dst = nullptr;

    cudaMalloc(&d_src, inputImage.total() * inputImage.elemSize());
    cudaMalloc(&d_dst, inputImage.total() * sizeof(Npp8u)); // 그레이스케일 이미지: 1 채널

    cudaMemcpy(d_src, inputImage.data, inputImage.total() * inputImage.elemSize(), cudaMemcpyHostToDevice);

    // NPP 계수 설정 (RGB를 그레이스케일로 변환하기 위한 기본 계수)
    Npp32f aCoeffs[3] = { 0.299f, 0.587f, 0.114f };

    // NPP 스트림 컨텍스트 생성 및 초기화
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = cudaStream;
    // nppStream 필드는 최신 NPP 라이브러리에서는 사용되지 않음

    // NPP 그레이스케일 변환 함수 호출
    NppiSize oSizeROI = { inputImage.cols, inputImage.rows };
    NppStatus status = nppiColorToGray_8u_C3C1R_Ctx(d_src, inputImage.cols * inputImage.elemSize(),
        d_dst, inputImage.cols,
        oSizeROI, aCoeffs, nppStreamCtx);

    if (status != NPP_SUCCESS) {
        std::cerr << "NPP 함수 호출 실패: " << status << std::endl;
        cudaFree(d_src);
        cudaFree(d_dst);
        cudaStreamDestroy(cudaStream);
        return cv::Mat();
    }

    // CUDA 스트림 동기화
    cudaStreamSynchronize(cudaStream);

    // 결과를 호스트 메모리로 복사
    cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC1);
    cudaMemcpy(outputImage.data, d_dst, outputImage.total() * outputImage.elemSize(), cudaMemcpyDeviceToHost);

    // CUDA 메모리 해제
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaStreamDestroy(cudaStream);

    return outputImage;
}

cv::Mat ImageProcessorNPP::rotate(cv::Mat& inputImage, bool isRight) {

    std::cout << "ImageProcessorNPP::rotate" << std::endl;

    if (inputImage.empty()) {
        std::cerr << "ImageProcessorNPP::rotate Input image is empty." << std::endl;
        return cv::Mat();
    }

    int srcWidth = inputImage.cols;
    int srcHeight = inputImage.rows;
    int numChannels = inputImage.channels();

    std::cout << "Input image size: " << srcWidth << " x " << srcHeight << std::endl;
    std::cout << "Number of channels: " << numChannels << std::endl;

    // 회전 각도 설정
    double angleInDegrees = isRight ? 270.0 : 90.0;
    double angleInRadians = angleInDegrees * CV_PI / 180.0; // 각도를 라디안 단위로 변환

    double cosAngle = std::abs(std::cos(angleInRadians));
    double sinAngle = std::abs(std::sin(angleInRadians));
    int dstWidth = static_cast<int>(srcWidth * cosAngle + srcHeight * sinAngle);
    int dstHeight = static_cast<int>(srcWidth * sinAngle + srcHeight * cosAngle);

    std::cout << "Cos(angle): " << cosAngle << std::endl;
    std::cout << "Sin(angle): " << sinAngle << std::endl;
    std::cout << "Calculated destination width: " << dstWidth << std::endl;
    std::cout << "Calculated destination height: " << dstHeight << std::endl;

    // 회전된 이미지를 위한 Mat 생성 (검정색 바탕)
    cv::Mat outputImage(dstHeight, dstWidth, inputImage.type(), cv::Scalar(0));

    std::cout << "Output image size: " << outputImage.cols << " x " << outputImage.rows << std::endl;

    // GPU 메모리로 이미지 업로드
    cv::cuda::GpuMat d_inputImage, d_outputImage;
    d_inputImage.upload(inputImage);
    d_outputImage.create(dstHeight, dstWidth, inputImage.type());

    std::cout << "GPU memory allocated for input and output images." << std::endl;
    std::cout << "Input image uploaded to GPU." << std::endl;
    std::cout << "Output image GPU Mat created with size: " << d_outputImage.cols << " x " << d_outputImage.rows << std::endl;

    // CUDA 스트림 생성
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::cout << "CUDA stream created." << std::endl;

    // NPP 함수 호출을 위한 포인터
    Npp8u* pSrc = d_inputImage.ptr<Npp8u>();
    Npp8u* pDst = d_outputImage.ptr<Npp8u>();

    std::cout << "Source GPU memory pointer: " << static_cast<void*>(pSrc) << std::endl;
    std::cout << "Destination GPU memory pointer: " << static_cast<void*>(pDst) << std::endl;

    // 이미지 크기 및 스트라이드
    NppiSize oSrcSize = { srcWidth, srcHeight };
    NppiSize oDstSize = { dstWidth, dstHeight };
    int srcStep = d_inputImage.step1();  // 입력 이미지의 행 간격
    int dstStep = d_outputImage.step1(); // 출력 이미지의 행 간격

    std::cout << "Source image size: " << oSrcSize.width << " x " << oSrcSize.height << std::endl;
    std::cout << "Destination image size: " << oDstSize.width << " x " << oDstSize.height << std::endl;
    std::cout << "Source step: " << srcStep << std::endl;
    std::cout << "Destination step: " << dstStep << std::endl;

    // 원본 이미지 중심 계산
    double srcCenterX = (srcWidth - 1) / 2.0;
    double srcCenterY = (srcHeight - 1) / 2.0;

    // 출력 이미지 중심 계산
    double dstCenterX = (dstWidth - 1) / 2.0;
    double dstCenterY = (dstHeight - 1) / 2.0;

    std::cout << "Source center: (" << srcCenterX << ", " << srcCenterY << ")" << std::endl;
    std::cout << "Destination center: (" << dstCenterX << ", " << dstCenterY << ")" << std::endl;

    // 회전 중심 조정
    double adjustedCenterX = dstCenterX;
    double adjustedCenterY = dstCenterY;

    // 출력 이미지의 ROI를 설정
    NppiRect oSrcROI = { 0, 0, srcWidth, srcHeight };
    NppiRect oDstROI = { 0, 0, dstWidth, dstHeight };

    std::cout << "Source ROI: x=" << oSrcROI.x << ", y=" << oSrcROI.y
        << ", width=" << oSrcROI.width << ", height=" << oSrcROI.height << std::endl;
    std::cout << "Destination ROI: x=" << oDstROI.x << ", y=" << oDstROI.y
        << ", width=" << oDstROI.width << ", height=" << oDstROI.height << std::endl;

    NppStatus nppStatus;
    if (numChannels == 3) {
        // 컬러 이미지 회전
        nppStatus = nppiRotate_8u_C3R(
            pSrc, oSrcSize, srcStep, oSrcROI,
            pDst, dstStep, oDstROI,
            angleInDegrees, adjustedCenterX, adjustedCenterY, NPPI_INTER_LINEAR
        );
    }
    else if (numChannels == 1) {
        // 그레이스케일 이미지 회전
        nppStatus = nppiRotate_8u_C1R(
            pSrc, oSrcSize, srcStep, oSrcROI,
            pDst, dstStep, oDstROI,
            angleInDegrees, adjustedCenterX, adjustedCenterY, NPPI_INTER_LINEAR
        );
    }
    else {
        std::cerr << "Unsupported image format." << std::endl;
        cudaStreamDestroy(stream);
        return cv::Mat();
    }

    // NPP 오류 처리
    if (nppStatus != NPP_SUCCESS) {
        std::cerr << "NPP Error: " << nppStatus << std::endl;
        cudaStreamDestroy(stream);
        return cv::Mat();
    }

    std::cout << "NPP function executed successfully." << std::endl;

    // GPU 이미지를 CPU 메모리로 다운로드
    d_outputImage.download(outputImage);

    // 디버그: 출력 이미지의 상위 왼쪽 코너 픽셀 값을 확인
    if (!outputImage.empty()) {
        std::cout << "Top-left corner pixel value: ";
        if (numChannels == 1) {
            std::cout << static_cast<int>(outputImage.at<uchar>(0, 0)) << std::endl;
        }
        else if (numChannels == 3) {
            cv::Vec3b pixel = outputImage.at<cv::Vec3b>(0, 0);
            std::cout << static_cast<int>(pixel[0]) << " "
                << static_cast<int>(pixel[1]) << " "
                << static_cast<int>(pixel[2]) << std::endl;
        }
    }

    std::cout << "Image downloaded from GPU to CPU." << std::endl;

    // CUDA 스트림 파괴
    cudaStreamDestroy(stream);
    std::cout << "CUDA stream destroyed." << std::endl;

    return outputImage;
   
    //cv::warpAffine 사용
    /* 
    std::cout << "ImageProcessorNPP::rotate" << std::endl;

    if (inputImage.empty()) {
        std::cerr << "ImageProcessorNPP::rotate Input image is empty." << std::endl;
        return cv::Mat();
    }

    int srcWidth = inputImage.cols;
    int srcHeight = inputImage.rows;
    int numChannels = inputImage.channels();

    std::cout << "Input image size: " << srcWidth << " x " << srcHeight << std::endl;
    std::cout << "Number of channels: " << numChannels << std::endl;

    // 회전 각도 설정
    double angleInDegrees = isRight ? 270.0 : 90.0;
    double angleInRadians = angleInDegrees * CV_PI / 180.0; // 각도를 라디안 단위로 변환

    double cosAngle = std::abs(std::cos(angleInRadians));
    double sinAngle = std::abs(std::sin(angleInRadians));
    int dstWidth = static_cast<int>(srcWidth * cosAngle + srcHeight * sinAngle);
    int dstHeight = static_cast<int>(srcWidth * sinAngle + srcHeight * cosAngle);

    std::cout << "Cos(angle): " << cosAngle << std::endl;
    std::cout << "Sin(angle): " << sinAngle << std::endl;
    std::cout << "Calculated destination width: " << dstWidth << std::endl;
    std::cout << "Calculated destination height: " << dstHeight << std::endl;

    // 회전된 이미지를 위한 Mat 생성 (검정색 바탕)
    cv::Mat outputImage(dstHeight, dstWidth, inputImage.type(), cv::Scalar(0));

    std::cout << "Output image size: " << outputImage.cols << " x " << outputImage.rows << std::endl;

    // 회전 행렬 계산
    cv::Point2f center(srcWidth / 2.0, srcHeight / 2.0);
    cv::Mat rotMat = cv::getRotationMatrix2D(center, angleInDegrees, 1.0);

    // 회전 행렬을 조정하여 출력 이미지의 중앙이 입력 이미지의 중앙과 맞도록 함
    rotMat.at<double>(0, 2) += (dstWidth / 2.0) - center.x;
    rotMat.at<double>(1, 2) += (dstHeight / 2.0) - center.y;

    // 이미지 회전
    cv::warpAffine(inputImage, outputImage, rotMat, outputImage.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

    // 디버그: 출력 이미지의 상위 왼쪽 코너 픽셀 값을 확인
    if (!outputImage.empty()) {
        std::cout << "Top-left corner pixel value: ";
        if (numChannels == 1) {
            std::cout << static_cast<int>(outputImage.at<uchar>(0, 0)) << std::endl;
        }
        else if (numChannels == 3) {
            cv::Vec3b pixel = outputImage.at<cv::Vec3b>(0, 0);
            std::cout << static_cast<int>(pixel[0]) << " "
                << static_cast<int>(pixel[1]) << " "
                << static_cast<int>(pixel[2]) << std::endl;
        }
    }

    std::cout << "Image rotation complete." << std::endl; */

    //return outputImage;
}

cv::Mat ImageProcessorNPP::zoom(cv::Mat& inputImage, double newWidth, double newHeight) {
    
    int inputWidth = inputImage.cols;
    int inputHeight = inputImage.rows;
    int newWidthInt = static_cast<int>(newWidth);
    int newHeightInt = static_cast<int>(newHeight);

    // 디버깅: 입력 이미지 크기 출력
    std::cout << "Input Image Size: " << inputWidth << "x" << inputHeight << std::endl;
    std::cout << "New Image Size: " << newWidthInt << "x" << newHeightInt << std::endl;

    // GPU 메모리로 이미지 업로드
    cv::cuda::GpuMat d_inputImage;
    d_inputImage.upload(inputImage);

    // GPU 메모리에 출력 이미지를 위한 GpuMat
    cv::cuda::GpuMat d_outputImage(newHeightInt, newWidthInt, inputImage.type());

    // NPP 함수 호출을 위한 포인터
    Npp8u* pSrc = d_inputImage.ptr<Npp8u>();
    Npp8u* pDst = d_outputImage.ptr<Npp8u>();

    // 이미지 크기 및 스트라이드
    NppiSize oSrcSize = { inputWidth, inputHeight };
    NppiSize oDstSize = { newWidthInt, newHeightInt };
    int srcStep = d_inputImage.step;  // 입력 이미지의 행 간격
    int dstStep = d_outputImage.step; // 출력 이미지의 행 간격

    // 디버깅: 스트라이드 값 출력
    std::cout << "Source Step: " << srcStep << std::endl;
    std::cout << "Destination Step: " << dstStep << std::endl;

    // 입력 및 출력 ROI 설정
    NppiRect oSrcRectROI = { 0, 0, inputWidth, inputHeight };
    NppiRect oDstRectROI = { 0, 0, newWidthInt, newHeightInt };

    // 디버깅: ROI 값 출력
    std::cout << "Source ROI: " << oSrcRectROI.x << ", " << oSrcRectROI.y << ", " << oSrcRectROI.width << ", " << oSrcRectROI.height << std::endl;
    std::cout << "Destination ROI: " << oDstRectROI.x << ", " << oDstRectROI.y << ", " << oDstRectROI.width << ", " << oDstRectROI.height << std::endl;

    NppStatus nppStatus;

    if (inputImage.channels() == 3) {
        // NPP 컬러 이미지 리사이즈 함수 호출
        nppStatus = nppiResize_8u_C3R(
            pSrc, srcStep,         // 입력 이미지의 stride
            oSrcSize,              // 입력 이미지 크기
            oSrcRectROI,           // 입력 이미지 영역
            pDst, dstStep,         // 출력 이미지의 stride
            oDstSize,              // 출력 이미지 크기
            oDstRectROI,           // 출력 이미지 영역
            NPPI_INTER_LINEAR      // 보간법 (선형 보간법)
        );
    }
    else if (inputImage.channels() == 1) {
        // NPP 그레이스케일 이미지 리사이즈 함수 호출
        nppStatus = nppiResize_8u_C1R(
            pSrc, srcStep,         // 입력 이미지의 stride
            oSrcSize,              // 입력 이미지 크기
            oSrcRectROI,           // 입력 이미지 영역
            pDst, dstStep,         // 출력 이미지의 stride
            oDstSize,              // 출력 이미지 크기
            oDstRectROI,           // 출력 이미지 영역
            NPPI_INTER_LINEAR      // 보간법 (선형 보간법)
        );
    }
    else {
        std::cerr << "Unsupported image format." << std::endl;
        return cv::Mat();
    }

    // NPP 오류 처리
    checkNPPError(nppStatus);

    // GPU 이미지를 CPU 메모리로 다운로드
    cv::Mat outputImage(newHeightInt, newWidthInt, inputImage.type());
    d_outputImage.download(outputImage);

    return outputImage;
}



cv::Mat ImageProcessorNPP::gaussianBlur(cv::Mat& inputImage, int kernelSize) {
    if (kernelSize % 2 == 0 || kernelSize < 3) {
        std::cerr << "Kernel size must be an odd number and greater than or equal to 3." << std::endl;
        return cv::Mat();
    }

    std::cout << "Kernel size: " << kernelSize << std::endl;
    std::cout << "Input image size: " << inputImage.cols << " x " << inputImage.rows << std::endl;
    std::cout << "Input image type: " << cv::typeToString(inputImage.type()) << std::endl;

    cv::Mat outputImage(inputImage.size(), inputImage.type());

    void* d_input = nullptr;
    void* d_output = nullptr;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc(&d_input, inputImage.rows * inputImage.cols * inputImage.elemSize());
    std::cout << "cudaMalloc for d_input: " << cudaGetErrorString(cudaStatus) << std::endl;
    std::cout << "d_input address: " << d_input << std::endl;
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for d_input. Error code: " << cudaStatus << std::endl;
        return cv::Mat();
    }

    cudaStatus = cudaMalloc(&d_output, inputImage.rows * inputImage.cols * inputImage.elemSize());
    std::cout << "cudaMalloc for d_output: " << cudaGetErrorString(cudaStatus) << std::endl;
    std::cout << "d_output address: " << d_output << std::endl;
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for d_output. Error code: " << cudaStatus << std::endl;
        cudaFree(d_input);
        return cv::Mat();
    }

    cudaStatus = cudaMemcpy(d_input, inputImage.data, inputImage.rows * inputImage.cols * inputImage.elemSize(), cudaMemcpyHostToDevice);
    std::cout << "cudaMemcpy HostToDevice: " << cudaGetErrorString(cudaStatus) << std::endl;
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to copy data to GPU. Error code: " << cudaStatus << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return cv::Mat();
    }

    NppiSize oSizeROI = { inputImage.cols, inputImage.rows };
    NppiMaskSize maskSize;

    switch (kernelSize) {
    case 3: maskSize = NPP_MASK_SIZE_3_X_3; break;
    case 5: maskSize = NPP_MASK_SIZE_5_X_5; break;
    case 7: maskSize = NPP_MASK_SIZE_7_X_7; break;
    default:
        std::cerr << "Unsupported mask size. Only 3, 5, and 7 are supported." << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return cv::Mat();
    }

    int srcStep = inputImage.cols * inputImage.elemSize();
    int dstStep = inputImage.cols * inputImage.elemSize();

    std::cout << "NPP Filter Gaussian parameters: " << std::endl;
    std::cout << "Source Step (stride): " << srcStep << std::endl;
    std::cout << "Destination Step (stride): " << dstStep << std::endl;
    std::cout << "ROI Size: " << oSizeROI.width << " x " << oSizeROI.height << std::endl;
    std::cout << "Mask Size: " << kernelSize << std::endl;

    // Check image type and select appropriate NPP function
    NppStatus status;
    switch (inputImage.type()) {
    case CV_8UC1: {
        status = nppiFilterGauss_8u_C1R(
            static_cast<Npp8u*>(d_input),
            inputImage.cols * inputImage.elemSize(),
            static_cast<Npp8u*>(d_output),
            inputImage.cols * inputImage.elemSize(),
            oSizeROI,
            maskSize
        );
        break;
    }
    case CV_16UC1: {
        status = nppiFilterGauss_16u_C1R(
            static_cast<Npp16u*>(d_input),
            inputImage.cols * inputImage.elemSize(),
            static_cast<Npp16u*>(d_output),
            inputImage.cols * inputImage.elemSize(),
            oSizeROI,
            maskSize
        );
        break;
    }
    case CV_16SC1: {
        status = nppiFilterGauss_16s_C1R(
            static_cast<Npp16s*>(d_input),
            inputImage.cols * inputImage.elemSize(),
            static_cast<Npp16s*>(d_output),
            inputImage.cols * inputImage.elemSize(),
            oSizeROI,
            maskSize
        );
        break;
    }
    case CV_8UC3: { // 16
        status = nppiFilterGauss_8u_C3R(
            static_cast<Npp8u*>(d_input),
            inputImage.cols * inputImage.elemSize(),
            static_cast<Npp8u*>(d_output),
            inputImage.cols * inputImage.elemSize(),
            oSizeROI,
            maskSize
        );
        break;
    }
    case CV_16UC3: {
        status = nppiFilterGauss_16u_C3R(
            static_cast<Npp16u*>(d_input),
            inputImage.cols * inputImage.elemSize(),
            static_cast<Npp16u*>(d_output),
            inputImage.cols * inputImage.elemSize(),
            oSizeROI,
            maskSize
        );
        break;
    }
    case CV_16SC3: {
        status = nppiFilterGauss_16s_C3R(
            static_cast<Npp16s*>(d_input),
            inputImage.cols * inputImage.elemSize(),
            static_cast<Npp16s*>(d_output),
            inputImage.cols * inputImage.elemSize(),
            oSizeROI,
            maskSize
        );
        break;
    }
    default: {
        std::cerr << "Unsupported image type. Type code: " << inputImage.type() << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return cv::Mat();
    }
    }


    std::cout << "nppiFilterGauss status: " << status << std::endl;
    if (status != NPP_SUCCESS) {
        std::cerr << "Failed to apply NPP Gaussian filter. Error code: " << status << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return cv::Mat();
    }

    cudaStatus = cudaMemcpy(outputImage.data, d_output, inputImage.rows * inputImage.cols * inputImage.elemSize(), cudaMemcpyDeviceToHost);
    std::cout << "cudaMemcpy DeviceToHost: " << cudaGetErrorString(cudaStatus) << std::endl;
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to copy result back to host. Error code: " << cudaStatus << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return cv::Mat();
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return outputImage;
}




cv::Mat ImageProcessorNPP::cannyEdges(cv::Mat& inputImage) {

    cv::Mat grayImage;
    cv::Mat edgeImage;
    cv::Mat outputImage;

    // Convert to grayscale if necessary
    if (inputImage.channels() == 3) {
        //cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
        grayImage = grayScale(inputImage);
    }
    else {
        grayImage = inputImage.clone();
    }

    // Prepare parameters for NPP function
    NppiSize oSrcSize = { grayImage.cols, grayImage.rows };
    NppiSize oSizeROI = { grayImage.cols, grayImage.rows };
    NppiPoint oSrcOffset = { 0, 0 };
    int srcStep = static_cast<int>(grayImage.step);
    int dstStep = srcStep;

    // Allocate GPU memory and upload data
    cv::cuda::GpuMat d_src, d_dst;
    d_src.upload(grayImage);
    d_dst.create(grayImage.size(), CV_8UC1);

    // Create NPP buffer
    int bufferSize = 0;
    NppStatus status = nppiFilterCannyBorderGetBufferSize(oSizeROI, &bufferSize);
    checkNPPStatus(status, "nppiFilterCannyBorderGetBufferSize");

    // Ensure that buffer size is valid
    if (bufferSize <= 0) {
        std::cerr << "Invalid buffer size: " << bufferSize << std::endl;
        return cv::Mat();
    }

    cv::cuda::GpuMat nppBuffer;
    nppBuffer.create(1, bufferSize, CV_8UC1);
    Npp8u* pBuffer = nppBuffer.ptr<Npp8u>();

    Npp8u* pSrcData = d_src.ptr<Npp8u>();
    Npp8u* pDstData = d_dst.ptr<Npp8u>();

    // Call the NPP function
    status = nppiFilterCannyBorder_8u_C1R(
        pSrcData, srcStep, oSrcSize, oSrcOffset,
        pDstData, dstStep, oSizeROI,
        NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3,
        50, 150, NppiNorm(nppiNormL2),
        NPP_BORDER_REPLICATE, pBuffer
    );
    checkNPPStatus(status, "nppiFilterCannyBorder_8u_C1R");

    // Download result from GPU
    d_dst.download(edgeImage);

    // Ensure edgeImage is correctly processed
    if (edgeImage.empty()) {
        std::cerr << "Edge image is empty after download." << std::endl;
        return cv::Mat();
    }

    if (inputImage.channels() == 3) {
        // 컬러 이미지 처리
        outputImage = inputImage.clone();
        for (int y = 0; y < edgeImage.rows; ++y) {
            for (int x = 0; x < edgeImage.cols; ++x) {
                if (edgeImage.at<uchar>(y, x) > 0) {
                    outputImage.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0); // 초록색
                }
            }
        }
    }
    else {
        // 흑백 이미지 처리
        outputImage = cv::Mat(grayImage.size(), CV_8UC1, cv::Scalar(0));
        for (int y = 0; y < edgeImage.rows; ++y) {
            for (int x = 0; x < edgeImage.cols; ++x) {
                if (edgeImage.at<uchar>(y, x) > 0) {
                    outputImage.at<uchar>(y, x) = 255; // 흰색
                }
            }
        }
    }

    return outputImage;
}

cv::Mat ImageProcessorNPP::medianFilter(cv::Mat& inputImage) {    

    if (inputImage.empty()) {
        std::cerr << "Input image is empty." << std::endl;
        return cv::Mat(); // 빈 이미지 반환
    }

    int numChannels = inputImage.channels();
    cv::Mat outputImage(inputImage.size(), inputImage.type());

    Npp32s nSrcStep = inputImage.step[0];
    Npp32s nDstStep = outputImage.step[0];
    NppiSize oSizeROI = { inputImage.cols, inputImage.rows };
    NppiSize oMaskSize = { 5, 5 }; // 5x5 필터
    NppiPoint oAnchor = { 2, 2 }; // 5x5 필터의 중앙

    try {
        NppStatus status;
        Npp32u nBufferSize = 0;

        // CUDA 초기화
        cudaSetDevice(0);        

        // 버퍼 메모리 할당 및 이미지 처리
        Npp8u* d_buffer;
        cudaMalloc(&d_buffer, nBufferSize);

        Npp8u* d_src;
        Npp8u* d_dst;
        size_t imageSize = inputImage.rows * inputImage.step;

        cudaMalloc(&d_src, imageSize);
        cudaMalloc(&d_dst, imageSize);

        // 입력 이미지를 GPU 메모리로 복사
        cudaMemcpy(d_src, inputImage.data, imageSize, cudaMemcpyHostToDevice);

        // 미디안 필터 적용
        if (numChannels == 3) {
            status = nppiFilterMedian_8u_C3R(
                d_src, nSrcStep, d_dst, nDstStep, oSizeROI, oMaskSize, oAnchor, d_buffer
            );

            std::cout << "nppiFilterMedian_8u_C3R status: " << status << std::endl;
            if (status != NPP_SUCCESS) {
                std::cerr << "nppiFilterMedian_8u_C3R failed with status: " << status << std::endl;
                throw std::runtime_error("nppiFilterMedian_8u_C3R failed.");
            }
        }
        else if (numChannels == 1) {
            status = nppiFilterMedian_8u_C1R(
                d_src, nSrcStep, d_dst, nDstStep, oSizeROI, oMaskSize, oAnchor, d_buffer
            );

            std::cout << "nppiFilterMedian_8u_C1R status: " << status << std::endl;
            if (status != NPP_SUCCESS) {
                std::cerr << "nppiFilterMedian_8u_C1R failed with status: " << status << std::endl;
                throw std::runtime_error("nppiFilterMedian_8u_C1R failed.");
            }
        }

        // 결과를 호스트로 복사
        cudaMemcpy(outputImage.data, d_dst, imageSize, cudaMemcpyDeviceToHost);

        // GPU 메모리 해제
        cudaFree(d_src);
        cudaFree(d_dst);
        cudaFree(d_buffer);

        // CUDA 에러 체크
        cudaError_t cudaErr = cudaGetLastError();
        if (cudaErr != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(cudaErr) << std::endl;
            throw std::runtime_error("CUDA error occurred.");
        }

    }
    catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return cv::Mat(); // 빈 이미지 반환
    }

    return outputImage;
}

cv::Mat ImageProcessorNPP::sobelFilter(cv::Mat& inputImage) {
    if (inputImage.empty()) {
        std::cerr << "Input image is empty." << std::endl;
        return cv::Mat(); // 빈 이미지 반환
    }

    int numChannels = inputImage.channels();
    cv::Mat outputImage(inputImage.size(), inputImage.type());

    Npp32s nSrcStep = inputImage.step[0];
    Npp32s nDstStep = outputImage.step[0];
    NppiSize oSizeROI = { inputImage.cols, inputImage.rows };

    NppStatus status;

    // CUDA 초기화
    cudaSetDevice(0);

    size_t imageSize = inputImage.rows * inputImage.step;

    unsigned char* d_src;
    unsigned char* d_dst;

    // CUDA 메모리 할당
    cudaMalloc(&d_src, imageSize);
    cudaMalloc(&d_dst, imageSize);

    // 입력 이미지를 GPU 메모리로 복사
    cudaMemcpy(d_src, inputImage.data, imageSize, cudaMemcpyHostToDevice);

    // 소벨 필터 처리
    try {
        if (numChannels == 3) {
            // 수평 소벨 필터
            status = nppiFilterSobelHoriz_8u_C3R(d_src, nSrcStep, d_dst, nDstStep, oSizeROI);
            if (status != NPP_SUCCESS) {
                std::cerr << "nppiFilterSobelHoriz_8u_C3R failed with status: " << status << std::endl;
                throw std::runtime_error("nppiFilterSobelHoriz_8u_C3R failed.");
            }

            // 수직 소벨 필터
            status = nppiFilterSobelVert_8u_C3R(d_dst, nSrcStep, d_dst, nDstStep, oSizeROI);
            if (status != NPP_SUCCESS) {
                std::cerr << "nppiFilterSobelVert_8u_C3R failed with status: " << status << std::endl;
                throw std::runtime_error("nppiFilterSobelVert_8u_C3R failed.");
            }
        }
        else if (numChannels == 1) {
            // 그레이스케일 이미지에서 소벨 필터 적용

            // 수평 소벨 필터
            status = nppiFilterSobelHoriz_8u_C1R(d_src, nSrcStep, d_dst, nDstStep, oSizeROI);
            if (status != NPP_SUCCESS) {
                std::cerr << "nppiFilterSobelHoriz_8u_C1R failed with status: " << status << std::endl;
                throw std::runtime_error("nppiFilterSobelHoriz_8u_C1R failed.");
            }

            // 수직 소벨 필터
            status = nppiFilterSobelVert_8u_C1R(d_dst, nSrcStep, d_dst, nDstStep, oSizeROI);
            if (status != NPP_SUCCESS) {
                std::cerr << "nppiFilterSobelVert_8u_C1R failed with status: " << status << std::endl;
                throw std::runtime_error("nppiFilterSobelVert_8u_C1R failed.");
            }
        }
        else {
            std::cerr << "Unsupported number of channels: " << numChannels << std::endl;
            cudaFree(d_src);
            cudaFree(d_dst);
            return cv::Mat(); // 빈 이미지 반환
        }

        // 결과를 호스트로 복사
        cudaMemcpy(outputImage.data, d_dst, imageSize, cudaMemcpyDeviceToHost);

        // CUDA 오류 체크
        cudaError_t cudaErr = cudaGetLastError();
        if (cudaErr != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(cudaErr) << std::endl;
            throw std::runtime_error("CUDA error occurred.");
        }
    }
    catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        cudaFree(d_src);
        cudaFree(d_dst);
        return cv::Mat(); // 빈 이미지 반환
    }

    // GPU 메모리 해제
    cudaFree(d_src);
    cudaFree(d_dst);

    return outputImage;
}

cv::Mat ImageProcessorNPP::laplacianFilter(cv::Mat& inputImage)
{
    std::cout << "ImageProcessorNPP::laplacianFilter" << std::endl;

    if (inputImage.empty()) {
        std::cerr << "Input image is empty." << std::endl;
        return cv::Mat();
    }

    int srcWidth = inputImage.cols;
    int srcHeight = inputImage.rows;
    int numChannels = inputImage.channels();

    // GPU 메모리로 이미지 업로드
    cv::cuda::GpuMat d_inputImage, d_outputImage;
    d_inputImage.upload(inputImage);

    // 라플라시안 필터 커널 정의
    Npp32s laplacianKernel3x3[9] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };

    NppiPoint oSrcOffset = { 0, 0 };
    NppiMaskSize eMaskSize = NPP_MASK_SIZE_5_X_5;
    NppiBorderType eBorderType = NPP_BORDER_REPLICATE; // 지원되는 경계 처리 방법

    NppiSize oSrcSize = { srcWidth, srcHeight };
    NppiSize oSizeROI = { srcWidth, srcHeight };

    int srcStep = d_inputImage.step;
    int dstStep = srcStep; // Destination step과 source step은 같음

    NppStatus nppStatus;

    cv::Mat outputImage; // Output image declaration
    outputImage.create(srcHeight, srcWidth, inputImage.type()); // Create the output image matrix

    if (numChannels == 1) {
        // 그레이스케일 이미지에 라플라시안 필터 적용
        d_outputImage.create(srcHeight, srcWidth, CV_8UC1); // Grayscale image type

        nppStatus = nppiFilterLaplaceBorder_8u_C1R(
            d_inputImage.ptr<Npp8u>(), srcStep,
            oSrcSize, oSrcOffset,
            d_outputImage.ptr<Npp8u>(), dstStep,
            oSizeROI, eMaskSize, eBorderType
        );

        if (nppStatus != NPP_SUCCESS) {
            std::cerr << "NPP Error: " << nppStatus << std::endl;
            return cv::Mat();
        }
    }
    else if (numChannels == 3) {
        // 컬러 이미지의 경우, 각 채널에 대해 필터 적용
        cv::cuda::GpuMat d_channels[3];
        cv::cuda::split(d_inputImage, d_channels);

        cv::cuda::GpuMat d_outputChannels[3];
        for (int i = 0; i < 3; ++i) {
            d_outputChannels[i].create(srcHeight, srcWidth, CV_8UC1);
        }

        for (int i = 0; i < 3; ++i) {
            nppStatus = nppiFilterLaplaceBorder_8u_C1R(
                d_channels[i].ptr<Npp8u>(), d_channels[i].step,
                oSrcSize, oSrcOffset,
                d_outputChannels[i].ptr<Npp8u>(), d_outputChannels[i].step,
                oSizeROI, eMaskSize, eBorderType
            );
            if (nppStatus != NPP_SUCCESS) {
                std::cerr << "NPP Error in channel " << i << ": " << nppStatus << std::endl;
                return cv::Mat();
            }
        }

        // 필터링된 채널을 병합하여 최종 출력 이미지 생성
        d_outputImage.create(srcHeight, srcWidth, CV_8UC3);
        std::vector<cv::cuda::GpuMat> channels(d_outputChannels, d_outputChannels + 3);
        cv::cuda::merge(channels, d_outputImage);

        // GPU 메모리에서 CPU 메모리로 다운로드
        try {
            d_outputImage.download(outputImage);
        }
        catch (const cv::Exception& e) {
            std::cerr << "OpenCV Exception during download: " << e.what() << std::endl;
            return cv::Mat();
        }

        return outputImage;
    }
    else {
        std::cerr << "Unsupported image format." << std::endl;
        return cv::Mat();
    }

    // 그레이스케일 이미지의 경우
    try {
        d_outputImage.download(outputImage);
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV Exception during download: " << e.what() << std::endl;
        return cv::Mat();
    }

    return outputImage;
}

cv::Mat ImageProcessorNPP::bilateralFilter(cv::Mat& inputImage)
{
    // 입력 이미지를 GPU 메모리로 복사
    Npp8u* d_inputImage;
    size_t inputImagePitch;
    cudaMallocPitch((void**)&d_inputImage, &inputImagePitch, inputImage.cols * sizeof(Npp8u) * inputImage.channels(), inputImage.rows);
    cudaMemcpy2D(d_inputImage, inputImagePitch, inputImage.data, inputImage.step, inputImage.cols * sizeof(Npp8u) * inputImage.channels(), inputImage.rows, cudaMemcpyHostToDevice);

    // 출력 이미지를 GPU 메모리로 할당
    cv::Mat outputImage(inputImage.size(), inputImage.type());
    Npp8u* d_outputImage;
    size_t outputImagePitch;
    cudaMallocPitch((void**)&d_outputImage, &outputImagePitch, outputImage.cols * sizeof(Npp8u) * outputImage.channels(), outputImage.rows);

    // 양방향 필터 파라미터 설정
    NppiSize oSrcSize = { inputImage.cols, inputImage.rows };
    NppiPoint oSrcOffset = { 0, 0 };
    Npp32f nValSquareSigma = 75.0f;
    Npp32f nPosSquareSigma = 75.0f;
    int nRadius = 9;
    NppiBorderType eBorderType = NPP_BORDER_REPLICATE;

    // 이미지 채널 수에 따라 함수 선택
    if (inputImage.channels() == 1) {
        NppStatus status = nppiFilterBilateralGaussBorder_8u_C1R(d_inputImage, static_cast<int>(inputImagePitch), oSrcSize, oSrcOffset,
            d_outputImage, static_cast<int>(outputImagePitch), oSrcSize,
            nRadius, 1, nValSquareSigma, nPosSquareSigma, eBorderType);
        if (status != NPP_SUCCESS) {
            std::cerr << "Error applying bilateral filter: " << status << std::endl;
        }
    }
    else if (inputImage.channels() == 3) {
        NppStatus status = nppiFilterBilateralGaussBorder_8u_C3R(d_inputImage, static_cast<int>(inputImagePitch), oSrcSize, oSrcOffset,
            d_outputImage, static_cast<int>(outputImagePitch), oSrcSize,
            nRadius, 3, nValSquareSigma, nPosSquareSigma, eBorderType);
        if (status != NPP_SUCCESS) {
            std::cerr << "Error applying bilateral filter: " << status << std::endl;
        }
    }
    else {
        std::cerr << "Unsupported number of channels: " << inputImage.channels() << std::endl;
    }

    // 처리된 이미지를 호스트로 복사
    cudaMemcpy2D(outputImage.data, outputImage.step, d_outputImage, outputImagePitch, outputImage.cols * sizeof(Npp8u) * outputImage.channels(), outputImage.rows, cudaMemcpyDeviceToHost);

    // 메모리 해제
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return outputImage;
}