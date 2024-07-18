#include "ImageProcessorNPP.h"

ImageProcessorNPP::ImageProcessorNPP()
{
}

ImageProcessorNPP::~ImageProcessorNPP()
{
}

cv::Mat ImageProcessorNPP::rotate(cv::Mat& inputImage, bool isRight) {
    // �Է� �̹��� ũ��
    int srcWidth = inputImage.cols;
    int srcHeight = inputImage.rows;

    // ȸ�� �� �̹��� ũ�� ����
    int dstWidth = srcHeight;
    int dstHeight = srcWidth;

    // ��� �̹����� ���� �޸� �Ҵ� (ȸ�� �� ũ��)
    cv::Mat outputImage(dstWidth, dstHeight, inputImage.type());

    // OpenCV Mat�� NPP ȣȯ �������� ��ȯ
    Npp8u* pSrc = inputImage.data;
    Npp8u* pDst = outputImage.data;
    NppiSize oSrcSize = { srcWidth, srcHeight };
    NppiRect oSrcROI = { 0, 0, srcWidth, srcHeight };
    int nSrcStep = srcWidth * inputImage.elemSize();
    int nDstStep = dstWidth * outputImage.elemSize();

    // ȸ�� ���� ���� (90��, �ð� ���� �Ǵ� �ݽð� ����)
    double angleInRadians = isRight ? -CV_PI / 2.0 : CV_PI / 2.0;

    // ��� �̹����� ROI�� ���� (���ο� ���ΰ� �ٲ� ����)
    NppiRect oDstROI = { 0, 0, dstWidth, dstHeight };

    // GPU �޸� �Ҵ�
    Npp8u* d_src = nullptr;
    Npp8u* d_dst = nullptr;
    cudaMalloc((void**)&d_src, srcWidth * srcHeight * inputImage.elemSize());
    cudaMalloc((void**)&d_dst, dstWidth * dstHeight * outputImage.elemSize());

    // �Է� �̹��� �����͸� GPU �޸𸮷� ����
    cudaMemcpy(d_src, pSrc, srcWidth * srcHeight * inputImage.elemSize(), cudaMemcpyHostToDevice);

    // NPP�� ����Ͽ� ȸ�� ����
    NppStatus status = nppiRotate_8u_C3R(d_src, oSrcSize, nSrcStep, oSrcROI,
        d_dst, nDstStep, oDstROI, angleInRadians,
        srcWidth / 2.0, srcHeight / 2.0, NPPI_INTER_LINEAR);

    if (status != NPP_SUCCESS) {
        std::cerr << "nppiRotate_8u_C3R error: " << status << std::endl;
        // GPU �޸� ����
        cudaFree(d_src);
        cudaFree(d_dst);
        return cv::Mat();
    }

    // ȸ�� �� GPU���� CPU�� ������ ����
    cudaMemcpy(pDst, d_dst, dstWidth * dstHeight * outputImage.elemSize(), cudaMemcpyDeviceToHost);

    // GPU �޸� ����
    cudaFree(d_src);
    cudaFree(d_dst);

    return outputImage;
}



cv::Mat ImageProcessorNPP::bilateralFilter(cv::Mat& inputImage)
{
    // �Է� �̹����� GPU �޸𸮷� ����
    Npp8u* d_inputImage;
    size_t inputImagePitch;
    cudaMallocPitch((void**)&d_inputImage, &inputImagePitch, inputImage.cols * sizeof(Npp8u) * inputImage.channels(), inputImage.rows);
    cudaMemcpy2D(d_inputImage, inputImagePitch, inputImage.data, inputImage.step, inputImage.cols * sizeof(Npp8u) * inputImage.channels(), inputImage.rows, cudaMemcpyHostToDevice);

    // ��� �̹����� GPU �޸𸮷� �Ҵ�
    cv::Mat outputImage(inputImage.size(), inputImage.type());
    Npp8u* d_outputImage;
    size_t outputImagePitch;
    cudaMallocPitch((void**)&d_outputImage, &outputImagePitch, outputImage.cols * sizeof(Npp8u) * outputImage.channels(), outputImage.rows);

    // ����� ���� �Ķ���� ����
    NppiSize oSrcSize = { inputImage.cols, inputImage.rows };
    NppiPoint oSrcOffset = { 0, 0 };
    Npp32f nValSquareSigma = 75.0f;
    Npp32f nPosSquareSigma = 75.0f;
    int nRadius = 9;
    NppiBorderType eBorderType = NPP_BORDER_REPLICATE;

    // �̹��� ä�� ���� ���� �Լ� ����
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

    // ó���� �̹����� ȣ��Ʈ�� ����
    cudaMemcpy2D(outputImage.data, outputImage.step, d_outputImage, outputImagePitch, outputImage.cols * sizeof(Npp8u) * outputImage.channels(), outputImage.rows, cudaMemcpyDeviceToHost);

    // �޸� ����
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return outputImage;
}