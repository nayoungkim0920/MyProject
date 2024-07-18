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

// 이미지의 픽셀 값을 출력하는 함수
void printPixelValues(const std::string& message, const cv::Mat& image) {
    std::cout << message << ": ";
    int numChannels = image.channels();
    for (int y = 0; y < std::min(image.rows, 1); ++y) {
        for (int x = 0; x < std::min(image.cols, 10); ++x) {
            const uchar* pixel = image.ptr<uchar>(y) + x * numChannels;
            std::cout << "(";
            for (int c = 0; c < numChannels; ++c) {
                std::cout << static_cast<int>(pixel[c]);
                if (c < numChannels - 1) std::cout << ", ";
            }
            std::cout << ") ";
        }
        std::cout << std::endl;
    }
}

cv::Mat ImageProcessorNPP::rotate(cv::Mat& inputImage, bool isRight) {
    int srcWidth = inputImage.cols;
    int srcHeight = inputImage.rows;
    int numChannels = inputImage.channels();

    // 회전된 이미지의 크기 계산
    double angleInDegrees = isRight ? 90.0 : -90.0;
    double angleInRadians = angleInDegrees * CV_PI / 180.0; // 각도를 라디안 단위로 변환

    double cosAngle = std::abs(std::cos(angleInRadians));
    double sinAngle = std::abs(std::sin(angleInRadians));
    int dstWidth = static_cast<int>(srcWidth * cosAngle + srcHeight * sinAngle);
    int dstHeight = static_cast<int>(srcWidth * sinAngle + srcHeight * cosAngle);

    // 회전된 이미지를 위한 Mat 생성
    cv::Mat outputImage(dstHeight, dstWidth, inputImage.type());

    Npp8u* pSrc = inputImage.data;
    Npp8u* pDst = outputImage.data;
    NppiSize oSrcSize = { srcWidth, srcHeight };
    NppiRect oSrcROI = { 0, 0, srcWidth, srcHeight };
    int nSrcStep = inputImage.step;
    int nDstStep = outputImage.step;

    // 회전 중심 설정 (이미지의 중심)
    double centerX = (srcWidth - 1) / 2.0;
    double centerY = (srcHeight - 1) / 2.0;

    // 출력 이미지의 ROI를 설정
    NppiRect oDstROI = { 0, 0, dstWidth, dstHeight };

    Npp8u* d_src = nullptr;
    Npp8u* d_dst = nullptr;

    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((void**)&d_src, nSrcStep * srcHeight);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc d_src error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return cv::Mat();
    }
    cudaStatus = cudaMalloc((void**)&d_dst, nDstStep * dstHeight);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc d_dst error: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_src);
        return cv::Mat();
    }
    cudaStatus = cudaMemset(d_dst, 0, nDstStep * dstHeight);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemset d_dst error: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_src);
        cudaFree(d_dst);
        return cv::Mat();
    }

    cudaStatus = cudaMemcpy(d_src, pSrc, nSrcStep * srcHeight, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy to d_src error: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_src);
        cudaFree(d_dst);
        return cv::Mat();
    }

    // 이미지 채널 수 확인 및 회전 수행
    NppStatus status;
    if (numChannels == 3) {
        // 컬러 이미지 회전
        status = nppiRotate_8u_C3R(d_src, oSrcSize, nSrcStep, oSrcROI,
            d_dst, nDstStep, oDstROI, angleInDegrees,
            centerX, centerY, NPPI_INTER_LINEAR);
    }
    else if (numChannels == 1) {
        // 그레이스케일 이미지 회전
        status = nppiRotate_8u_C1R(d_src, oSrcSize, nSrcStep, oSrcROI,
            d_dst, nDstStep, oDstROI, angleInDegrees,
            centerX, centerY, NPPI_INTER_LINEAR);
    }
    else {
        std::cerr << "Unsupported image format." << std::endl;
        cudaFree(d_src);
        cudaFree(d_dst);
        return cv::Mat();
    }

    if (status != NPP_SUCCESS) {
        std::cerr << "nppiRotate error: " << status << std::endl;
        cudaFree(d_src);
        cudaFree(d_dst);
        return cv::Mat();
    }

    // GPU에서 CPU로 데이터 복사
    cudaStatus = cudaMemcpy(pDst, d_dst, nDstStep * dstHeight, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy to pDst error: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_src);
        cudaFree(d_dst);
        return cv::Mat();
    }

    cudaFree(d_src);
    cudaFree(d_dst);

    return outputImage;
}

cv::Mat ImageProcessorNPP::gaussianBlur(cv::Mat& inputImage, int kernelSize) {

    if (kernelSize % 2 == 0 || kernelSize < 3) {
        std::cerr << "Kernel size must be an odd number and greater than or equal to 3." << std::endl;
        return cv::Mat();
    }

    std::cout << "Kernel size: " << kernelSize << std::endl;

    cv::Mat outputImage(inputImage.size(), inputImage.type());

    std::cout << "Input image size: " << inputImage.cols << " x " << inputImage.rows << std::endl;
    std::cout << "Input image type: " << inputImage.type() << std::endl;

    void* d_input = nullptr;
    void* d_output = nullptr;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc(&d_input, inputImage.rows * inputImage.cols * inputImage.elemSize());
    std::cout << "cudaMalloc for d_input: " << cudaGetErrorString(cudaStatus) << std::endl;
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for d_input. Error code: " << cudaStatus << std::endl;
        return cv::Mat();
    }

    cudaStatus = cudaMalloc(&d_output, inputImage.rows * inputImage.cols * inputImage.elemSize());
    std::cout << "cudaMalloc for d_output: " << cudaGetErrorString(cudaStatus) << std::endl;
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
    NppiMaskSize maskSize = (NppiMaskSize)kernelSize;

    std::cout << "NPP Filter Gaussian parameters: " << std::endl;
    std::cout << "ROI Size: " << oSizeROI.width << " x " << oSizeROI.height << std::endl;
    std::cout << "Mask Size: " << maskSize << std::endl;

    NppStatus status;
    switch (inputImage.type()) {
    case CV_8UC3: {
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