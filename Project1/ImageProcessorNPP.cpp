#include "ImageProcessorNPP.h"

ImageProcessorNPP::ImageProcessorNPP()
{
}

ImageProcessorNPP::~ImageProcessorNPP()
{
}

cv::Mat ImageProcessorNPP::rotate(cv::Mat& inputImage, bool isRight) {
    // 입력 이미지 크기
    int srcWidth = inputImage.cols;
    int srcHeight = inputImage.rows;

    // 회전 후 이미지 크기 설정
    int dstWidth = srcHeight;
    int dstHeight = srcWidth;

    // 출력 이미지를 위한 메모리 할당 (회전 후 크기)
    cv::Mat outputImage(dstWidth, dstHeight, inputImage.type());

    // OpenCV Mat을 NPP 호환 형식으로 변환
    Npp8u* pSrc = inputImage.data;
    Npp8u* pDst = outputImage.data;
    NppiSize oSrcSize = { srcWidth, srcHeight };
    NppiRect oSrcROI = { 0, 0, srcWidth, srcHeight };
    int nSrcStep = srcWidth * inputImage.elemSize();
    int nDstStep = dstWidth * outputImage.elemSize();

    // 회전 각도 설정 (90도, 시계 방향 또는 반시계 방향)
    double angleInRadians = isRight ? -CV_PI / 2.0 : CV_PI / 2.0;

    // 출력 이미지의 ROI를 설정 (가로와 세로가 바뀜에 주의)
    NppiRect oDstROI = { 0, 0, dstWidth, dstHeight };

    // GPU 메모리 할당
    Npp8u* d_src = nullptr;
    Npp8u* d_dst = nullptr;
    cudaMalloc((void**)&d_src, srcWidth * srcHeight * inputImage.elemSize());
    cudaMalloc((void**)&d_dst, dstWidth * dstHeight * outputImage.elemSize());

    // 입력 이미지 데이터를 GPU 메모리로 복사
    cudaMemcpy(d_src, pSrc, srcWidth * srcHeight * inputImage.elemSize(), cudaMemcpyHostToDevice);

    // NPP를 사용하여 회전 수행
    NppStatus status = nppiRotate_8u_C3R(d_src, oSrcSize, nSrcStep, oSrcROI,
        d_dst, nDstStep, oDstROI, angleInRadians,
        srcWidth / 2.0, srcHeight / 2.0, NPPI_INTER_LINEAR);

    if (status != NPP_SUCCESS) {
        std::cerr << "nppiRotate_8u_C3R error: " << status << std::endl;
        // GPU 메모리 해제
        cudaFree(d_src);
        cudaFree(d_dst);
        return cv::Mat();
    }

    // 회전 후 GPU에서 CPU로 데이터 복사
    cudaMemcpy(pDst, d_dst, dstWidth * dstHeight * outputImage.elemSize(), cudaMemcpyDeviceToHost);

    // GPU 메모리 해제
    cudaFree(d_src);
    cudaFree(d_dst);

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