#include "ImageProcessorIPP.h"

ImageProcessorIPP::ImageProcessorIPP()
{
}

ImageProcessorIPP::~ImageProcessorIPP()
{
}

cv::Mat ImageProcessorIPP::rotate(cv::Mat& inputImage, bool isRight)
{
    std::cout << "Input image size: " << inputImage.cols << " x " << inputImage.rows << std::endl;

    double angle; //90.0 ������, 270.0 ����

    if (isRight)
        angle = 90.0;
    else
        angle = 270.0;

    // Input image size
    IppiSize srcSize = { inputImage.cols, inputImage.rows };

    // Convert rotation angle to radians
    double angleRadians = angle * CV_PI / 180.0;
    std::cout << "Rotation angle (radians): " << angleRadians << std::endl;

    // Set the size of the output image after rotation
    double cosAngle = std::abs(std::cos(angleRadians));
    double sinAngle = std::abs(std::sin(angleRadians));
    int dstWidth = static_cast<int>(srcSize.width * cosAngle + srcSize.height * sinAngle);
    int dstHeight = static_cast<int>(srcSize.width * sinAngle + srcSize.height * cosAngle);
    IppiSize dstSize = { dstWidth, dstHeight };
    std::cout << "Output image size after rotation: " << dstWidth << " x " << dstHeight << std::endl;

    // Create an output image that can contain the entire rotated image
    cv::Mat outputImage(dstSize.height, dstSize.width, inputImage.type());
    std::cout << "Output image created" << std::endl;

    // Affine transform coefficients for IPP
    double xShift = static_cast<double>(srcSize.width) / 2.0;  // x shift: half the width of the image
    double yShift = static_cast<double>(srcSize.height) / 2.0; // y shift: half the height of the image
    std::cout << "xShift: " << xShift << ", yShift: " << yShift << std::endl;

    // Calculate the affine transform coefficients based on direction
    double coeffs[2][3];
    coeffs[0][0] = std::cos(angleRadians);
    coeffs[0][1] = -std::sin(angleRadians);
    coeffs[0][2] = xShift - xShift * std::cos(angleRadians) + yShift * std::sin(angleRadians) + (dstWidth - srcSize.width) / 2.0;
    coeffs[1][0] = std::sin(angleRadians);
    coeffs[1][1] = std::cos(angleRadians);
    coeffs[1][2] = yShift - xShift * std::sin(angleRadians) - yShift * std::cos(angleRadians) + (dstHeight - srcSize.height) / 2.0;
    
    std::cout << "Affine transform coefficients calculated" << std::endl;

    // Variables needed for IPP
    IppiWarpSpec* pSpec = nullptr;
    Ipp8u* pBuffer = nullptr;
    int specSize = 0, initSize = 0, bufSize = 0;
    const Ipp32u numChannels = 3; // Number of channels in the input image (BGR)
    IppiBorderType borderType = ippBorderConst;
    Ipp64f pBorderValue[numChannels];
    for (int i = 0; i < numChannels; ++i) pBorderValue[i] = 255.0;
    std::cout << "pBorderValue set" << std::endl;

    // Set the sizes of the spec and init buffers
    IppStatus status;
    status = ippiWarpAffineGetSize(srcSize, dstSize, ipp8u, coeffs, ippLinear, ippWarpForward, borderType, &specSize, &initSize);
    
    if (status != ippStsNoErr) {
        std::cerr << "ippiWarpAffineGetSize error: " << status << std::endl;
        return cv::Mat();
    }
    std::cout << "ippiWarpAffineGetSize completed, specSize: " << specSize << ", initSize: " << initSize << std::endl;

    // Memory allocation
    pSpec = (IppiWarpSpec*)ippsMalloc_8u(specSize);
    if (pSpec == nullptr) {
        std::cerr << "Memory allocation error for pSpec" << std::endl;
        return cv::Mat();
    }
    std::cout << "pSpec memory allocation completed" << std::endl;

    // Filter initialization
    status = ippiWarpAffineLinearInit(srcSize, dstSize, ipp8u, coeffs, ippWarpForward, numChannels, borderType, pBorderValue, 0, pSpec);
    
    if (status != ippStsNoErr)
    {
        std::cerr << "ippiWarpAffineLinearInit error: " << status << std::endl;
        ippsFree(pSpec);
        return cv::Mat();
    }
    std::cout << "ippiWarpAffineLinearInit completed" << std::endl;

    // Get work buffer size
    status = ippiWarpGetBufferSize(pSpec, dstSize, &bufSize);
    if (status != ippStsNoErr) {
        std::cerr << "ippiWarpGetBufferSize error: " << status << std::endl;
        ippsFree(pSpec);
        return cv::Mat();
    }
    std::cout << "ippiWarpGetBufferSize completed, bufSize: " << bufSize << std::endl;

    pBuffer = ippsMalloc_8u(bufSize);
    if (pBuffer == nullptr) {
        std::cerr << "Memory allocation error for pBuffer" << std::endl;
        ippsFree(pSpec);
        return cv::Mat();
    }
    std::cout << "pBuffer memory allocation completed" << std::endl;

    // Define dstOffset (ensure it's non-negative and within the image bounds)
    IppiPoint dstOffset = { 0, 0 };
    std::cout << "dstOffset set, x: " << dstOffset.x << ", y: " << dstOffset.y << std::endl;

    // Rotate the image using IPP
    status = ippiWarpAffineLinear_8u_C3R(inputImage.data, inputImage.step, outputImage.data, outputImage.step,
            dstOffset, dstSize, pSpec, pBuffer);
    
    if (status != ippStsNoErr) {
        std::cerr << "ippiWarpAffineLinear_8u_C3R error: " << status << std::endl;
        ippsFree(pSpec);
        ippsFree(pBuffer);
        return cv::Mat();
    }
    std::cout << "Image rotation completed" << std::endl;

    // Output the size of the processed image
    std::cout << "Output image size: " << outputImage.cols << " x " << outputImage.rows << std::endl;

    // Free memory
    ippsFree(pSpec);
    ippsFree(pBuffer);
    std::cout << "Memory freed" << std::endl;

    return outputImage;
}

cv::Mat ImageProcessorIPP::grayScale(cv::Mat& inputImage)
{    
    ippInit();

    // �Է� �̹����� ũ�� �� ���� ����
    IppiSize roiSize = { inputImage.cols, inputImage.rows };
    int srcStep = inputImage.step;
    int dstStep = inputImage.cols;
    Ipp8u* srcData = inputImage.data;

    // ��� �̹��� ���� �� IPP �޸� �Ҵ�
    cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC1);
    Ipp8u* dstData = outputImage.data;

    // IPP RGB to Gray ��ȯ ����
    IppStatus status = ippiRGBToGray_8u_C3C1R(srcData, srcStep, dstData, dstStep, roiSize);
    if (status != ippStsNoErr) {
        std::cerr << "IPP ����: " << status << std::endl;
        return cv::Mat(); // ���� �߻� �� �� Mat ��ȯ
    }    

    return outputImage;
}

cv::Mat ImageProcessorIPP::zoom(cv::Mat& inputImage, int newWidth, int newHeight)
{
    // IPP ������ ����
    IppStatus status;
    IppiSize srcSize = { inputImage.cols, inputImage.rows };
    IppiSize dstSize = { static_cast<int>(newWidth), static_cast<int>(newHeight) };
    IppiPoint dstOffset = { 0, 0 };
    std::vector<Ipp8u> pBuffer;
    IppiResizeSpec_32f* pSpec = nullptr;

    // ũ�� �� �ʱ�ȭ ���� �Ҵ�
    int specSize = 0, initSize = 0, bufSize = 0;
    status = ippiResizeGetSize_8u(srcSize, dstSize, ippNearest, 0, &specSize, &initSize);
    if (status != ippStsNoErr) {
        std::cerr << "Error: ippiResizeGetSize_8u failed with status code " << status << std::endl;
        return cv::Mat();
    }

    pSpec = (IppiResizeSpec_32f*)(ippMalloc(specSize));
    if (!pSpec) {
        std::cerr << "Error: Memory allocation failed for pSpec" << std::endl;
        return cv::Mat();
    }

    pBuffer.resize(initSize);
    if (pBuffer.empty()) {
        std::cerr << "Error: Memory allocation failed for pBuffer" << std::endl;
        ippFree(pSpec);
        return cv::Mat();
    }

    // ũ�� ���� ���� �ʱ�ȭ
    status = ippiResizeNearestInit_8u(srcSize, dstSize, pSpec);
    if (status != ippStsNoErr) {
        std::cerr << "Error: ippiResizeNearestInit_8u failed with status code " << status << std::endl;
        ippFree(pSpec);
        return cv::Mat();
    }

    // Get the size of the working buffer
    status = ippiResizeGetBufferSize_8u(pSpec, dstSize, inputImage.channels(), &bufSize);
    if (status != ippStsNoErr) {
        std::cerr << "Error: ippiResizeGetBufferSize_8u failed with status code " << status << std::endl;
        ippFree(pSpec);
        return cv::Mat();
    }

    pBuffer.resize(bufSize);
    if (pBuffer.empty()) {
        std::cerr << "Error: Memory allocation failed for pBuffer" << std::endl;
        ippFree(pSpec);
        return cv::Mat();
    }

    // ũ�� ���� ����
    cv::Mat outputImage(dstSize.height, dstSize.width, inputImage.type());
    Ipp8u* pSrcData = reinterpret_cast<Ipp8u*>(inputImage.data);
    Ipp8u* pDstData = reinterpret_cast<Ipp8u*>(outputImage.data);

    // �̹��� Ÿ�Կ� ���� IPP �Լ� ȣ��
    if (inputImage.type() == CV_8UC3) {
        std::cerr << "ippiResizeNearest_8u_C3R" << std::endl;
        status = ippiResizeNearest_8u_C3R(pSrcData, inputImage.step[0], pDstData, outputImage.step[0], dstOffset, dstSize, pSpec, pBuffer.data());
    }
    else if (inputImage.type() == CV_16UC3) {
        std::cerr << "ippiResizeNearest_16u_C3R" << std::endl;
        status = ippiResizeNearest_16u_C3R(reinterpret_cast<Ipp16u*>(pSrcData), inputImage.step[0], reinterpret_cast<Ipp16u*>(pDstData), outputImage.step[0], dstOffset, dstSize, pSpec, pBuffer.data());
    }
    else if (inputImage.type() == CV_32FC3) {
        std::cerr << "ippiResizeNearest_32f_C3R" << std::endl;
        status = ippiResizeNearest_32f_C3R(reinterpret_cast<Ipp32f*>(pSrcData), inputImage.step[0], reinterpret_cast<Ipp32f*>(pDstData), outputImage.step[0], dstOffset, dstSize, pSpec, pBuffer.data());
    }
    else if (inputImage.type() == CV_8UC1) {
        std::cerr << "ippiResizeNearest_8u_C1R" << std::endl;
        status = ippiResizeNearest_8u_C1R(pSrcData, inputImage.step[0], pDstData, outputImage.step[0], dstOffset, dstSize, pSpec, pBuffer.data());
    }
    else if (inputImage.type() == CV_16UC1) {
        std::cerr << "ippiResizeNearest_16u_C1R" << std::endl;
        status = ippiResizeNearest_16u_C1R(reinterpret_cast<Ipp16u*>(pSrcData), inputImage.step[0], reinterpret_cast<Ipp16u*>(pDstData), outputImage.step[0], dstOffset, dstSize, pSpec, pBuffer.data());
    }
    else if (inputImage.type() == CV_32FC1) {
        std::cerr << "ippiResizeNearest_32f_C1R" << std::endl;
        status = ippiResizeNearest_32f_C1R(reinterpret_cast<Ipp32f*>(pSrcData), inputImage.step[0], reinterpret_cast<Ipp32f*>(pDstData), outputImage.step[0], dstOffset, dstSize, pSpec, pBuffer.data());
    }
    else if (inputImage.type() == CV_16SC1) {
        std::cerr << "ippiResizeNearest_16s_C1R" << std::endl;
        status = ippiResizeNearest_16s_C1R(reinterpret_cast<Ipp16s*>(pSrcData), inputImage.step[0], reinterpret_cast<Ipp16s*>(pDstData), outputImage.step[0], dstOffset, dstSize, pSpec, pBuffer.data());
    }
    else if (inputImage.type() == CV_16SC3) {
        std::cerr << "ippiResizeNearest_16s_C3R" << std::endl;
        status = ippiResizeNearest_16s_C3R(reinterpret_cast<Ipp16s*>(pSrcData), inputImage.step[0], reinterpret_cast<Ipp16s*>(pDstData), outputImage.step[0], dstOffset, dstSize, pSpec, pBuffer.data());
    }
    else {
        std::cerr << "Error: Unsupported image type" << std::endl;
        ippFree(pSpec);
        return cv::Mat();
    }

    if (status != ippStsNoErr) {
        std::cerr << "Error: ippiResizeNearest_8u failed with status code " << status << std::endl;
        ippFree(pSpec);
        return cv::Mat();
    }

    // �޸� ����
    ippFree(pSpec);

    return outputImage;
}

cv::Mat ImageProcessorIPP::gaussianBlur(cv::Mat& inputImage, int kernelSize)
{
    // �Է� �̹����� 3ä�� BGR �̹������� Ȯ���ϰ�, �ƴ� ��� ��ȯ
    cv::Mat bgrInputImage;
    if (inputImage.channels() != 3 || inputImage.type() != CV_8UC3) {
        if (inputImage.channels() == 1) {
            cv::cvtColor(inputImage, bgrInputImage, cv::COLOR_GRAY2BGR);
        }
        else {
            std::cerr << "Error: Unsupported image format." << std::endl;
            return cv::Mat();
        }
    }
    else {
        bgrInputImage = inputImage.clone(); // �̹� BGR�� ��� �״�� ���
    }

    // ��� �̹����� 16��Ʈ 3ä��(CV_16UC3)�� ����
    cv::Mat tempOutputImage(bgrInputImage.size(), CV_16UC3);

    // IPP �Լ��� ������ �����͵�
    Ipp16u* pSrc = reinterpret_cast<Ipp16u*>(bgrInputImage.data);
    Ipp16u* pDst = reinterpret_cast<Ipp16u*>(tempOutputImage.data);

    // ROI ũ�� ����
    IppiSize roiSize = { bgrInputImage.cols, bgrInputImage.rows };

    // ���͸��� ���� ���� �� ����Ʈ�� ũ�� ���
    int specSize, bufferSize;
    IppStatus status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp16u, 3, &specSize, &bufferSize);
    if (status != ippStsNoErr) {
        std::cerr << "Error: ippiFilterGaussianGetBufferSize failed with status " << status << std::endl;
        return cv::Mat(); // �� ��� ��ȯ
    }

    // �ܺ� ���� �Ҵ�
    Ipp8u* pBuffer = ippsMalloc_8u(bufferSize);
    if (pBuffer == nullptr) {
        std::cerr << "Error: Failed to allocate buffer." << std::endl;
        return cv::Mat(); // �� ��� ��ȯ
    }

    // ����þ� ���� ����Ʈ�� ����ü �޸� �Ҵ�
    IppFilterGaussianSpec* pSpec = reinterpret_cast<IppFilterGaussianSpec*>(ippsMalloc_8u(specSize));
    if (pSpec == nullptr) {
        std::cerr << "Error: Failed to allocate spec structure." << std::endl;
        ippsFree(pBuffer);
        return cv::Mat(); // �� ��� ��ȯ
    }

    // ����þ� ���� �ʱ�ȭ (ǥ�� ������ ���÷� 1.5�� ����)
    float sigma = 1.5f;
    status = ippiFilterGaussianInit(roiSize, kernelSize, sigma, ippBorderRepl, ipp16u, 3, pSpec, pBuffer);
    if (status != ippStsNoErr) {
        std::cerr << "Error: ippiFilterGaussianInit failed with status " << status << std::endl;
        ippsFree(pBuffer);
        ippsFree(pSpec);
        return cv::Mat(); // �� ��� ��ȯ
    }

    // ����þ� ���� ����
    int srcStep = bgrInputImage.cols * sizeof(Ipp16u) * 3;
    int dstStep = tempOutputImage.cols * sizeof(Ipp16u) * 3;
    Ipp16u borderValue[3] = { 0, 0, 0 }; // ����þ� ���� ���� �� ����� ���� ��
    status = ippiFilterGaussianBorder_16u_C3R(pSrc, srcStep, pDst, dstStep, roiSize, borderValue, pSpec, pBuffer);
    if (status != ippStsNoErr) {
        std::cerr << "Error: ippiFilterGaussianBorder_16u_C3R failed with status " << status << std::endl;
        ippsFree(pBuffer);
        ippsFree(pSpec);
        return cv::Mat(); // �� ��� ��ȯ
    }

    // �޸� ����
    ippsFree(pBuffer);
    ippsFree(pSpec);

    // 8��Ʈ �̹����� ��ȯ
    cv::Mat outputImage;
    tempOutputImage.convertTo(outputImage, CV_8UC3, 1.0 / 256.0);

    return outputImage;
}

cv::Mat ImageProcessorIPP::cannyEdges(cv::Mat& inputImage)
{
    cv::Mat grayImage;
    if (inputImage.channels() == 3) {
        grayImage = grayScale(inputImage);
    }
    else {
        grayImage = inputImage.clone();
    }

    // IPP�� ����Ͽ� Canny ���� ���� ����
    IppiSize roiSize = { grayImage.cols, grayImage.rows };
    int srcStep = grayImage.step;
    int dstStep = grayImage.cols;
    Ipp8u* srcData = grayImage.data;
    Ipp8u* dstData = ippsMalloc_8u(roiSize.width * roiSize.height); // ��� �̹��� �޸� �Ҵ�

    if (!dstData) {
        std::cerr << "Memory allocation error: Failed to allocate dstData" << std::endl;
        return cv::Mat(); // �޸� �Ҵ� ���� ó�� �ߴ�
    }

    // IPP Canny ó���� ���� �ӽ� ���� ũ�� ���
    int bufferSize;
    IppStatus status = ippiCannyBorderGetSize(roiSize, ippFilterSobel, ippMskSize3x3, ipp8u, &bufferSize);
    if (status != ippStsNoErr) {
        std::cerr << "IPP error: Failed to calculate buffer size for Canny edge detection (" << status << ")";
        ippsFree(dstData); // �Ҵ�� �޸� ����
        return cv::Mat(); // ���� �߻� �� ó�� �ߴ�
    }

    // �ӽ� ���� �Ҵ�
    Ipp8u* pBuffer = ippsMalloc_8u(bufferSize);
    if (!pBuffer) {
        std::cerr << "Memory allocation error: Failed to allocate dstData" << std::endl;
        ippsFree(dstData); // �̹� �Ҵ�� dstData �޸𸮵� ����
        return cv::Mat(); // �޸� �Ҵ� ���� ó�� �ߴ�
    }

    // IPP Canny ���� ���� ����
    status = ippiCannyBorder_8u_C1R(srcData, srcStep, dstData, dstStep, roiSize, ippFilterSobel, ippMskSize3x3, ippBorderRepl, 0, 50.0f, 150.0f, ippNormL2, pBuffer);
    if (status != ippStsNoErr) {
        std::cerr << "IPP error: Failed to perform Canny edge detection (" << status << ")";
        ippsFree(pBuffer); // �Ҵ�� �޸� ����
        ippsFree(dstData); // �Ҵ�� �޸� ����
        return cv::Mat(); // ���� �߻� �� ó�� �ߴ�
    }

    // ����� OpenCV Mat �������� ��ȯ
    cv::Mat outputImage(grayImage.rows, grayImage.cols, CV_8UC1, dstData);

    // �Ҵ�� �޸� ����
    ippsFree(pBuffer);
    ippsFree(dstData);

    return outputImage;
}

Ipp8u* ImageProcessorIPP::matToIpp8u(cv::Mat& mat)
{
    return mat.ptr<Ipp8u>();
}


cv::Mat ImageProcessorIPP::medianFilter(cv::Mat& inputImage)
{
    // �Է� �̹����� �׷��̽��������� Ȯ��
    bool isGrayScale = (inputImage.channels() == 1);

    // IPP �̵�� ���� ����
    IppiSize roiSize = { inputImage.cols, inputImage.rows };
    IppiSize kernelSize = { 5, 5 }; // 5x5 Ŀ�� ũ��
    int bufferSize = 0;

    // IPP �ʱ�ȭ
    ippInit();

    // ���� ũ�� ���
    IppStatus status;
    if (isGrayScale) {
        status = ippiFilterMedianBorderGetBufferSize(roiSize, kernelSize, ipp8u, 1, &bufferSize);
    }
    else {
        status = ippiFilterMedianBorderGetBufferSize(roiSize, kernelSize, ipp8u, 3, &bufferSize);
    }

    if (status != ippStsNoErr) {
        std::cerr << "Error: ippiFilterMedianBorderGetBufferSize failed with status " << status << std::endl;
        return cv::Mat(); // �� ��� ��ȯ
    }

    Ipp8u* pBuffer = ippsMalloc_8u(bufferSize);

    // ��� �̹��� �ʱ�ȭ
    cv::Mat outputImage = cv::Mat::zeros(inputImage.size(), inputImage.type());

    // �̵�� ���� ����
    if (isGrayScale) {
        status = ippiFilterMedianBorder_8u_C1R(matToIpp8u(inputImage), inputImage.step[0], matToIpp8u(outputImage), outputImage.step[0], roiSize, kernelSize, ippBorderRepl, 0, pBuffer);
    }
    else {
        status = ippiFilterMedianBorder_8u_C3R(reinterpret_cast<Ipp8u*>(inputImage.data), inputImage.step[0], reinterpret_cast<Ipp8u*>(outputImage.data), outputImage.step[0], roiSize, kernelSize, ippBorderRepl, 0, pBuffer);
    }

    if (status != ippStsNoErr) {
        std::cerr << "Error: ippiFilterMedianBorder_8u_C1R or _8u_C3R failed with status " << status << std::endl;
        ippsFree(pBuffer);
        return cv::Mat(); // �� ��� ��ȯ
    }

    // �޸� ����
    ippsFree(pBuffer);

    return outputImage;
}

cv::Mat ImageProcessorIPP::laplacianFilter(cv::Mat& inputImage)
{
    // �Է� �̹����� IPP �������� ��ȯ
    IppiSize roiSize = { inputImage.cols, inputImage.rows };
    int step = inputImage.step;
    Ipp8u* pSrc = inputImage.data;

    // ��� �̹��� �غ�
    cv::Mat outputImage(inputImage.size(), CV_16S); // 16��Ʈ ���������� ����
    Ipp16s* pDst = reinterpret_cast<Ipp16s*>(outputImage.data); // ������ ����ȯ
    int dstStep = outputImage.step;

    // �ʿ��� ���� ũ�� ���
    int bufferSize = 0;
    IppStatus status = ippiFilterLaplacianGetBufferSize_8u16s_C1R(roiSize, ippMskSize3x3, &bufferSize);
    if (status != ippStsNoErr) {
        std::cerr << "Failed to get buffer size with status: " << status << std::endl;
        return cv::Mat();
    }

    Ipp8u* pBuffer = ippsMalloc_8u(bufferSize);

    // ���� ����
    status = ippiFilterLaplacianBorder_8u16s_C1R(
        pSrc, step, pDst, dstStep, roiSize, ippMskSize3x3, ippBorderRepl, 0, pBuffer);

    ippsFree(pBuffer);

    if (status != ippStsNoErr) {
        std::cerr << "IPP Laplacian filter failed with status: " << status << std::endl;
        return cv::Mat(); // ���� ó��
    }

    // ����� 8��Ʈ �̹����� ��ȯ
    outputImage.convertTo(outputImage, CV_8U);

    return outputImage;
}

cv::Mat ImageProcessorIPP::bilateralFilter(cv::Mat& inputImage)
{
    // IPP ����ü�� ���� ũ�� ���
    IppiSize roiSize = { inputImage.cols, inputImage.rows };
    int kernelSize = 9; // ���� ũ��
    IppDataType dataType = ipp8u; // ������ Ÿ�� (8��Ʈ ��ä�� �̹���)
    int numChannels = 3; // ä�� ��
    IppiDistanceMethodType distMethod = ippDistNormL2; // �Ÿ� ���
    int specSize = 0, bufferSize = 0;

    ippiFilterBilateralGetBufferSize(ippiFilterBilateralGauss, roiSize, kernelSize, dataType, numChannels, distMethod, &specSize, &bufferSize);

    // �޸� �Ҵ�
    IppiFilterBilateralSpec* pSpec = (IppiFilterBilateralSpec*)ippMalloc(specSize);
    Ipp8u* pBuffer = (Ipp8u*)ippMalloc(bufferSize);

    // ���� �ʱ�ȭ
    IppStatus status = ippiFilterBilateralInit(ippiFilterBilateralGauss, roiSize, kernelSize, dataType, numChannels, distMethod, 75, 75, pSpec);
    if (status != ippStsNoErr) {
        std::cerr << "Error initializing bilateral filter: " << status << std::endl;
        ippFree(pSpec);
        ippFree(pBuffer);
        return cv::Mat();
    }

    // ��� �̹��� ���� �Ҵ�
    cv::Mat outputImage(inputImage.size(), inputImage.type());

    // ����� ���� ����
    status = ippiFilterBilateral_8u_C3R(inputImage.ptr<Ipp8u>(), inputImage.step, outputImage.ptr<Ipp8u>(), outputImage.step, roiSize, ippBorderRepl, NULL, pSpec, pBuffer);
    if (status != ippStsNoErr) {
        std::cerr << "Error applying bilateral filter: " << status << std::endl;
    }

    // �޸� ����
    ippFree(pSpec);
    ippFree(pBuffer);

    return outputImage;

    // NPP
    /*
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

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

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "bilateralFilter", "NPP", elapsedTimeMs);

    // �޸� ����
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return result;
    */
}

cv::Mat ImageProcessorIPP::sobelFilter(cv::Mat& inputImage)
{
    // Convert input image to grayscale if it is not
    cv::Mat grayImage;
    if (inputImage.channels() > 1) {
        grayImage = grayScale(inputImage);  // Use OpenCV for conversion
    }
    else {
        grayImage = inputImage.clone();  // If already grayscale, make a copy
    }

    // Check if the grayImage is valid
    if (grayImage.empty()) {
        std::cerr << "Gray image is empty." << std::endl;
        return cv::Mat();
    }

    // Prepare destination image
    cv::Mat outputImage = cv::Mat::zeros(grayImage.size(), CV_16SC1);

    // IPP specific variables
    IppiSize roiSize = { grayImage.cols, grayImage.rows };
    IppiMaskSize mask = ippMskSize3x3; // Using 3x3 Sobel kernel
    IppNormType normType = ippNormL1;
    int bufferSize = 0;
    IppStatus status;

    // Calculate buffer size
    status = ippiFilterSobelGetBufferSize(roiSize, mask, normType, ipp8u, ipp16s, 1, &bufferSize);
    if (status != ippStsNoErr) {
        std::cerr << "Error calculating buffer size: " << status << std::endl;
        return cv::Mat();
    }

    // Allocate buffer
    Ipp8u* pBuffer = ippsMalloc_8u(bufferSize);
    if (!pBuffer) {
        std::cerr << "Error allocating buffer." << std::endl;
        return cv::Mat();
    }

    // Apply Sobel filter
    status = ippiFilterSobel_8u16s_C1R(
        grayImage.data,
        static_cast<int>(grayImage.step),
        reinterpret_cast<Ipp16s*>(outputImage.data),
        static_cast<int>(outputImage.step),
        roiSize,
        mask,
        normType,
        ippBorderRepl,
        0,
        pBuffer
    );

    if (status != ippStsNoErr) {
        std::cerr << "Error applying Sobel filter: " << status << std::endl;
        ippsFree(pBuffer);
        return cv::Mat();
    }

    // Free the buffer
    ippsFree(pBuffer);

    // Debug: Check if outputImage is valid
    if (outputImage.empty()) {
        std::cerr << "Output image is empty." << std::endl;
        return cv::Mat();
    }

    // Convert to absolute values and scale to 8-bit for visualization
    cv::convertScaleAbs(outputImage, outputImage);

    // Debug: Print minimum and maximum values of absOutputImage
    double minVal, maxVal;
    cv::minMaxLoc(outputImage, &minVal, &maxVal);
    std::cout << "Min value: " << minVal << ", Max value: " << maxVal << std::endl;

    return outputImage;
}