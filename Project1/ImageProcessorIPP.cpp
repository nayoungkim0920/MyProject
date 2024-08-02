#include "ImageProcessorIPP.h"

ImageProcessorIPP::ImageProcessorIPP()
{
}

ImageProcessorIPP::~ImageProcessorIPP()
{
}

cv::Mat ImageProcessorIPP::rotate(cv::Mat& inputImage, bool isRight) {
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
    IppiBorderType borderType = ippBorderConst;
    Ipp64f pBorderValue[4]; // 4 for up to 4 channels (RGBA)
    for (int i = 0; i < 4; ++i) pBorderValue[i] = 255.0;
    std::cout << "pBorderValue set" << std::endl;

    // Set the sizes of the spec and init buffers
    IppStatus status;
    int numChannels = inputImage.channels();
    if (numChannels == 3) {
        // For color image (BGR)
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

        if (status != ippStsNoErr) {
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

        // Rotate the image using IPP
        status = ippiWarpAffineLinear_8u_C3R(inputImage.data, inputImage.step, outputImage.data, outputImage.step,
            IppiPoint{ 0, 0 }, dstSize, pSpec, pBuffer);

        if (status != ippStsNoErr) {
            std::cerr << "ippiWarpAffineLinear_8u_C3R error: " << status << std::endl;
            ippsFree(pSpec);
            ippsFree(pBuffer);
            return cv::Mat();
        }
        std::cout << "Color image rotation completed" << std::endl;
    }
    else if (numChannels == 1) {
        // For grayscale image
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

        if (status != ippStsNoErr) {
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

        // Rotate the image using IPP
        status = ippiWarpAffineLinear_8u_C1R(inputImage.data, inputImage.step, outputImage.data, outputImage.step,
            IppiPoint{ 0, 0 }, dstSize, pSpec, pBuffer);

        if (status != ippStsNoErr) {
            std::cerr << "ippiWarpAffineLinear_8u_C1R error: " << status << std::endl;
            ippsFree(pSpec);
            ippsFree(pBuffer);
            return cv::Mat();
        }
        std::cout << "Grayscale image rotation completed" << std::endl;
    }
    else {
        std::cerr << "Unsupported image format." << std::endl;
        return cv::Mat();
    }

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

    // �Է� �̹����� ä�� �� Ȯ��
    int numChannels = inputImage.channels();

    // �̹� �׷��̽����� �̹����� ���
    if (numChannels == 1) {
        // �׷��̽����� �̹����� �״�� ��ȯ
        return inputImage.clone();
    }

    // �Է� �̹����� �÷��� ���
    if (numChannels != 3) {
        std::cerr << "Unsupported image format for grayscale conversion." << std::endl;
        return cv::Mat(); // �������� �ʴ� ������ ��� �� Mat ��ȯ
    }

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
    /*
    ipptypes.h
    typedef enum {
    ippNearest = IPPI_INTER_NN,
    ippLinear = IPPI_INTER_LINEAR,
    ippCubic = IPPI_INTER_CUBIC2P_CATMULLROM,
    ippLanczos = IPPI_INTER_LANCZOS,
    ippHahn = 0,
    ippSuper = IPPI_INTER_SUPER
    } IppiInterpolationType;
    */

    // IPP ������ ����
    IppStatus status;
    IppiSize srcSize = { inputImage.cols, inputImage.rows };
    IppiSize dstSize = { static_cast<int>(newWidth), static_cast<int>(newHeight) };
    IppiPoint dstOffset = { 0, 0 };
    std::vector<Ipp8u> pBuffer;
    IppiResizeSpec_32f* pSpec = nullptr;

    // ũ�� �� �ʱ�ȭ ���� �Ҵ�
    int specSize = 0, initSize = 0, bufSize = 0;
    //ippiResizeNearest -> aliasing�߻�
    //status = ippiResizeGetSize_8u(srcSize, dstSize, ippNearest, 0, &specSize, &initSize);
    status = ippiResizeGetSize_8u(srcSize, dstSize, ippLinear, 0, &specSize, &initSize);
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
    //ippiResizeNearest -> aliasing�߻�
    //status = ippiResizeNearestInit_8u(srcSize, dstSize, pSpec);
    status = ippiResizeLinearInit_8u(srcSize, dstSize, pSpec);
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

    //ippiResizeNearest -> aliasing�߻�
    if (inputImage.type() == CV_8UC3) { 
        //std::cerr << "ippiResizeNearest_8u_C3R" << std::endl;
        //status = ippiResizeNearest_8u_C3R(pSrcData, inputImage.step[0], pDstData, outputImage.step[0], dstOffset, dstSize, pSpec, pBuffer.data());
        std::cerr << "ippiResizeLinear_8u_C3R" << std::endl;
        status = ippiResizeLinear_8u_C3R(
            pSrcData,                    // Source data
            inputImage.step[0],          // Source step
            pDstData,                    // Destination data
            outputImage.step[0],         // Destination step
            dstOffset,                   // Destination offset
            dstSize,                     // Destination size
            ippBorderRepl,               // Border type
            nullptr,                     // Border value
            pSpec,                       // Spec structure
            pBuffer.data()               // Work buffer
        );
    }
    else if (inputImage.type() == CV_16UC3) {
        status = ippiResizeLinear_16u_C3R(
            reinterpret_cast<Ipp16u*>(pSrcData), inputImage.step[0],
            reinterpret_cast<Ipp16u*>(pDstData), outputImage.step[0],
            dstOffset, dstSize, ippBorderRepl, nullptr, pSpec, pBuffer.data());
    }
    else if (inputImage.type() == CV_32FC3) {
        status = ippiResizeLinear_32f_C3R(
            reinterpret_cast<Ipp32f*>(pSrcData), inputImage.step[0],
            reinterpret_cast<Ipp32f*>(pDstData), outputImage.step[0],
            dstOffset, dstSize, ippBorderRepl, nullptr, pSpec, pBuffer.data());
    }
    else if (inputImage.type() == CV_8UC1) {
        status = ippiResizeLinear_8u_C1R(
            pSrcData, inputImage.step[0],
            pDstData, outputImage.step[0],
            dstOffset, dstSize, ippBorderRepl, nullptr, pSpec, pBuffer.data());
    }
    else if (inputImage.type() == CV_16UC1) {
        status = ippiResizeLinear_16u_C1R(
            reinterpret_cast<Ipp16u*>(pSrcData), inputImage.step[0],
            reinterpret_cast<Ipp16u*>(pDstData), outputImage.step[0],
            dstOffset, dstSize, ippBorderRepl, nullptr, pSpec, pBuffer.data());
    }
    else if (inputImage.type() == CV_32FC1) {
        status = ippiResizeLinear_32f_C1R(
            reinterpret_cast<Ipp32f*>(pSrcData), inputImage.step[0],
            reinterpret_cast<Ipp32f*>(pDstData), outputImage.step[0],
            dstOffset, dstSize, ippBorderRepl, nullptr, pSpec, pBuffer.data());
    }
    else if (inputImage.type() == CV_16SC1) {
        status = ippiResizeLinear_16s_C1R(
            reinterpret_cast<Ipp16s*>(pSrcData), inputImage.step[0],
            reinterpret_cast<Ipp16s*>(pDstData), outputImage.step[0],
            dstOffset, dstSize, ippBorderRepl, nullptr, pSpec, pBuffer.data());
    }
    else if (inputImage.type() == CV_16SC3) {
        status = ippiResizeLinear_16s_C3R(
            reinterpret_cast<Ipp16s*>(pSrcData), inputImage.step[0],
            reinterpret_cast<Ipp16s*>(pDstData), outputImage.step[0],
            dstOffset, dstSize, ippBorderRepl, nullptr, pSpec, pBuffer.data());
    }
    else {
        std::cerr << "Error: Unsupported image type" << std::endl;
        ippFree(pSpec);
        return cv::Mat();
    }

    //Nearest
    /*
    * // �̹��� Ÿ�Կ� ���� IPP �Լ� ȣ��
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

    */
    
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
    std::cout << "Starting Gaussian Blur Processing..." << std::endl;

    cv::Mat bgrInputImage;
    cv::Mat tempOutputImage;

    // �Է� �̹����� �׷��̽��������� �÷������� ���� ó��
    if (inputImage.channels() == 1) {
        // �׷��̽����� �̹���
        bgrInputImage = inputImage.clone();
        std::cout << "Grayscale image detected." << std::endl;

        // �ӽ� ��� �̹����� 8��Ʈ �׷��̽����Ϸ� ����
        tempOutputImage.create(bgrInputImage.size(), CV_8UC1);

        // IPP ������ Ÿ�� ����
        Ipp8u* pSrc = bgrInputImage.data;
        Ipp8u* pDst = tempOutputImage.data;

        // ROI ũ�� ����
        IppiSize roiSize = { bgrInputImage.cols, bgrInputImage.rows };

        // Gaussian ���͸� ���� ���� �� ���� ������ ���
        int specSize = 0, bufferSize = 0;
        IppStatus status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp8u, 1, &specSize, &bufferSize);
        if (status != ippStsNoErr) {
            std::cerr << "Error: ippiFilterGaussianGetBufferSize failed with status " << status << std::endl;
            return cv::Mat();
        }

        Ipp8u* pBuffer = ippsMalloc_8u(bufferSize);
        if (pBuffer == nullptr) {
            std::cerr << "Error: Failed to allocate buffer." << std::endl;
            return cv::Mat();
        }

        IppFilterGaussianSpec* pSpec = reinterpret_cast<IppFilterGaussianSpec*>(ippsMalloc_8u(specSize));
        if (pSpec == nullptr) {
            std::cerr << "Error: Failed to allocate spec structure." << std::endl;
            ippsFree(pBuffer);
            return cv::Mat();
        }

        std::cout << "Debug Info:" << std::endl;
        std::cout << "ROI Size: (" << roiSize.width << ", " << roiSize.height << ")" << std::endl;
        std::cout << "Kernel Size: " << kernelSize << std::endl;
        std::cout << "Sigma: " << 1.5 << std::endl;
        std::cout << "Border Type: " << ippBorderRepl << std::endl;
        std::cout << "Data Type: " << ipp8u << std::endl;
        std::cout << "Number of Channels: " << 1 << std::endl;
        std::cout << "Spec Size: " << specSize << std::endl;
        std::cout << "Buffer Size: " << bufferSize << std::endl;

        // Gaussian ���� �ʱ�ȭ
        status = ippiFilterGaussianInit(roiSize, kernelSize, 1.5, ippBorderRepl, ipp8u, 1, pSpec, pBuffer);
        if (status != ippStsNoErr) {
            std::cerr << "Error: ippiFilterGaussianInit failed with status " << status << std::endl;
            ippsFree(pBuffer);
            ippsFree(pSpec);
            return cv::Mat();
        }

        // �ҽ��� ������ �̹����� �ܰ� ���
        int srcStep = bgrInputImage.cols * sizeof(Ipp8u);
        int dstStep = tempOutputImage.cols * sizeof(Ipp8u);

        // Gaussian ���� ����
        Ipp8u borderValue = 0;
        try {
            status = ippiFilterGaussian_8u_C1R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderRepl, &borderValue, pSpec, pBuffer);
            if (status != ippStsNoErr) {
                std::cerr << "Error: ippiFilterGaussian_8u_C1R failed with status " << status << std::endl;
                ippsFree(pBuffer);
                ippsFree(pSpec);
                throw std::runtime_error("ippiFilterGaussian_8u_C1R failed with status " + std::to_string(status));
            }
        }
        catch (const std::runtime_error& e) {
            std::cerr << "Runtime Error: " << e.what() << std::endl;
            return cv::Mat();
        }

        // �޸� ����
        ippsFree(pBuffer);
        ippsFree(pSpec);

        // ó���� �̹����� ��ȯ
        return tempOutputImage;
    }
    else if (inputImage.channels() == 3) {
        // �÷� �̹���
        bgrInputImage = inputImage.clone();
        std::cout << "Color image detected." << std::endl;

        // �ӽ� ��� �̹����� 8��Ʈ �÷��� ����
        tempOutputImage.create(bgrInputImage.size(), CV_8UC3);

        // IPP ������ Ÿ�� ����
        Ipp8u* pSrc = bgrInputImage.data;
        Ipp8u* pDst = tempOutputImage.data;

        // ROI ũ�� ����
        IppiSize roiSize = { bgrInputImage.cols, bgrInputImage.rows };

        // Gaussian ���͸� ���� ���� �� ���� ������ ���
        int specSize = 0, bufferSize = 0;
        IppStatus status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp8u, 3, &specSize, &bufferSize);
        if (status != ippStsNoErr) {
            std::cerr << "Error: ippiFilterGaussianGetBufferSize failed with status " << status << std::endl;
            return cv::Mat();
        }

        Ipp8u* pBuffer = ippsMalloc_8u(bufferSize);
        if (pBuffer == nullptr) {
            std::cerr << "Error: Failed to allocate buffer." << std::endl;
            return cv::Mat();
        }

        IppFilterGaussianSpec* pSpec = reinterpret_cast<IppFilterGaussianSpec*>(ippsMalloc_8u(specSize));
        if (pSpec == nullptr) {
            std::cerr << "Error: Failed to allocate spec structure." << std::endl;
            ippsFree(pBuffer);
            return cv::Mat();
        }

        // Gaussian ���� �ʱ�ȭ
        status = ippiFilterGaussianInit(roiSize, kernelSize, 1.5, ippBorderRepl, ipp8u, 3, pSpec, pBuffer);
        if (status != ippStsNoErr) {
            std::cerr << "Error: ippiFilterGaussianInit failed with status " << status << std::endl;
            ippsFree(pBuffer);
            ippsFree(pSpec);
            return cv::Mat();
        }

        // �ҽ��� ������ �̹����� �ܰ� ���
        int srcStep = bgrInputImage.cols * sizeof(Ipp8u) * 3;
        int dstStep = tempOutputImage.cols * sizeof(Ipp8u) * 3;

        std::cout << "Debug Info for Gaussian Filter:" << std::endl;
        std::cout << "Source Step: " << srcStep << std::endl;
        std::cout << "Destination Step: " << dstStep << std::endl;
        std::cout << "ROI Size: (" << roiSize.width << ", " << roiSize.height << ")" << std::endl;
        std::cout << "Border Type: " << ippBorderRepl << std::endl;
        std::cout << "Kernel Size: " << kernelSize << std::endl;
        std::cout << "Sigma: 1.5" << std::endl;
        std::cout << "Number of Channels: 3" << std::endl;
        std::cout << "Buffer Size: " << bufferSize << std::endl;
        std::cout << "Spec Size: " << specSize << std::endl;

        Ipp8u borderValue[3] = { 0, 0, 0 };
        try {
            status = ippiFilterGaussian_8u_C3R(pSrc, srcStep, pDst, dstStep, roiSize, ippBorderRepl, borderValue, pSpec, pBuffer);
            if (status != ippStsNoErr) {
                std::cerr << "Error: ippiFilterGaussian_8u_C3R failed with status " << status << std::endl;
                ippsFree(pBuffer);
                ippsFree(pSpec);
                throw std::runtime_error("ippiFilterGaussian_8u_C3R failed with status " + std::to_string(status));
            }
        }
        catch (const std::runtime_error& e) {
            std::cerr << "Runtime Error: " << e.what() << std::endl;
            return cv::Mat();
        }

        std::cout << "Border Values: (" << borderValue[0] << ", " << borderValue[1] << ", " << borderValue[2] << ")" << std::endl;

        // �޸� ����
        ippsFree(pBuffer);
        ippsFree(pSpec);

        // ó���� �̹����� ��ȯ
        return tempOutputImage;
    }
    else {
        std::cerr << "Error: Unsupported image format." << std::endl;
        return cv::Mat();
    }
}

cv::Mat ImageProcessorIPP::cannyEdges(cv::Mat& inputImage)
{
    cv::Mat grayImage;
    cv::Mat outputImage;

    if (inputImage.channels() == 3) {
        // �÷� �̹����� ���, �׷��̽����Ϸ� ��ȯ�Ͽ� Canny ���� ���� ����
        grayImage = grayScale(inputImage);
    }
    else {
        // ��� �̹����� ���
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
        std::cerr << "IPP error: Failed to calculate buffer size for Canny edge detection (" << status << ")" << std::endl;
        ippsFree(dstData); // �Ҵ�� �޸� ����
        return cv::Mat(); // ���� �߻� �� ó�� �ߴ�
    }

    // �ӽ� ���� �Ҵ�
    Ipp8u* pBuffer = ippsMalloc_8u(bufferSize);
    if (!pBuffer) {
        std::cerr << "Memory allocation error: Failed to allocate pBuffer" << std::endl;
        ippsFree(dstData); // �̹� �Ҵ�� dstData �޸𸮵� ����
        return cv::Mat(); // �޸� �Ҵ� ���� ó�� �ߴ�
    }

    // IPP Canny ���� ���� ����
    status = ippiCannyBorder_8u_C1R(srcData, srcStep, dstData, dstStep, roiSize, ippFilterSobel, ippMskSize3x3, ippBorderRepl, 0, 50.0f, 150.0f, ippNormL2, pBuffer);
    if (status != ippStsNoErr) {
        std::cerr << "IPP error: Failed to perform Canny edge detection (" << status << ")" << std::endl;
        ippsFree(pBuffer); // �Ҵ�� �޸� ����
        ippsFree(dstData); // �Ҵ�� �޸� ����
        return cv::Mat(); // ���� �߻� �� ó�� �ߴ�
    }

    // ����� OpenCV Mat �������� ��ȯ
    cv::Mat edgeImage(grayImage.rows, grayImage.cols, CV_8UC1, dstData);

    if (inputImage.channels() == 3) {
        // �÷� �̹����� ��, ������ �ʷϻ����� ǥ��
        outputImage = inputImage.clone();
        for (int y = 0; y < edgeImage.rows; ++y) {
            for (int x = 0; x < edgeImage.cols; ++x) {
                if (edgeImage.at<uchar>(y, x) > 0) {
                    outputImage.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0); // �ʷϻ�
                }
            }
        }
    }
    else {
        // ��� �̹����� ��
        outputImage = edgeImage;
    }

    // �Ҵ�� �޸� ����
    ippsFree(pBuffer);
    ippsFree(dstData);

    return outputImage;
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
    std::cout << __func__ << std::endl;

    int numChannels = inputImage.channels();
    cv::Mat grayImage;

    if (numChannels == 1) {
        grayImage = inputImage.clone();
    }
    else if (numChannels == 3) {
        grayImage = grayScale(inputImage);
    }
    else {
        std::cerr << __func__ << " : Unsupported number of channels: " << numChannels << std::endl;
        return cv::Mat(); // �� �̹��� ��ȯ
    }

    // Helper function to apply Laplacian filter to a single channel
    auto applyFilterToChannel = [](const cv::Mat& inputChannel, cv::Mat& outputChannel) {
        IppiSize roiSize = { inputChannel.cols, inputChannel.rows };
        int step = inputChannel.step;
        Ipp8u* pSrc = inputChannel.data;

        outputChannel.create(inputChannel.size(), CV_16S);
        Ipp16s* pDst = reinterpret_cast<Ipp16s*>(outputChannel.data);
        int dstStep = outputChannel.step;

        int bufferSize = 0;
        IppStatus status = ippiFilterLaplacianGetBufferSize_8u16s_C1R(roiSize, ippMskSize3x3, &bufferSize);
        if (status != ippStsNoErr) {
            std::cerr << "Failed to get buffer size with status: " << status << std::endl;
            return false;
        }

        Ipp8u* pBuffer = ippsMalloc_8u(bufferSize);
        status = ippiFilterLaplacianBorder_8u16s_C1R(
            pSrc, step, pDst, dstStep, roiSize, ippMskSize3x3, ippBorderRepl, 0, pBuffer
        );

        ippsFree(pBuffer);

        if (status != ippStsNoErr) {
            std::cerr << "IPP Laplacian filter failed with status: " << status << std::endl;
            return false;
        }

        return true;
    };

    cv::Mat outputImage;

    if (numChannels == 1) {
        // Apply filter to grayscale image
        bool success = applyFilterToChannel(inputImage, outputImage);
        if (!success) {
            return cv::Mat();
        }
        outputImage.convertTo(outputImage, CV_8U);
    }
    else if (numChannels == 3) {
        // Apply Laplacian filter to grayscale image
        cv::Mat grayLaplacian;
        bool success = applyFilterToChannel(grayImage, grayLaplacian);
        if (!success) {
            return cv::Mat();
        }
        grayLaplacian.convertTo(grayLaplacian, CV_8U);

        // Convert the Laplacian result to color
        cv::Mat coloredEdgeImage(grayImage.size(), CV_8UC3);
        cv::cvtColor(grayLaplacian, coloredEdgeImage, cv::COLOR_GRAY2BGR);

        // Overlay the Laplacian result on the original color image
        cv::Mat colorLaplacian;
        cv::addWeighted(inputImage, 0.5, coloredEdgeImage, 0.5, 0, colorLaplacian);
        outputImage = colorLaplacian;
    }

    return outputImage;

}

cv::Mat ImageProcessorIPP::bilateralFilter(cv::Mat& inputImage)
{
    int numChannels = inputImage.channels();
    IppiSize roiSize = { inputImage.cols, inputImage.rows };
    int kernelSize = 9; // Filter size
    IppDataType dataType;
    IppiDistanceMethodType distMethod; // Distance method
    int specSize = 0, bufferSize = 0;

    // Determine data type and allocate buffer size based on channels
    if (numChannels == 1) {
        dataType = ipp8u; // Grayscale
        distMethod = ippDistNormL1;

        // Calculate buffer size needed for the filter
        IppStatus status = ippiFilterBilateralBorderGetBufferSize(
            ippiFilterBilateralGauss, // Filter type
            roiSize,
            kernelSize / 2, // Radius of circular neighborhood
            dataType,
            numChannels,
            distMethod,
            &specSize,
            &bufferSize
        );

        std::cout << "Buffer size calculation status: " << status << std::endl;
        std::cout << "Calculated Spec size: " << specSize << ", Calculated Buffer size: " << bufferSize << std::endl;

        if (status != ippStsNoErr) {
            std::cerr << "Error getting buffer size: " << status << std::endl;
            switch (status) {
            case ippStsBadArgErr:
                std::cerr << "Bad argument error. Check input parameters." << std::endl;
                break;
            case ippStsSizeErr:
                std::cerr << "Size error. Check roiSize and kernelSize." << std::endl;
                break;
            case ippStsNotSupportedModeErr:
                std::cerr << "Not supported mode error. Check filter or distMethod." << std::endl;
                break;
            case ippStsDataTypeErr:
                std::cerr << "Data type error. Check dataType." << std::endl;
                break;
            case ippStsNumChannelsErr:
                std::cerr << "Number of channels error. Check numChannels." << std::endl;
                break;
            default:
                std::cerr << "Other error: " << status << std::endl;
                break;
            }
            return cv::Mat(); // Return empty image
        }

        // Allocate memory
        IppiFilterBilateralSpec* pSpec = (IppiFilterBilateralSpec*)ippMalloc(specSize);
        Ipp8u* pBuffer = (Ipp8u*)ippMalloc(bufferSize);
        std::cout << "Allocated memory for spec size: " << specSize << ", buffer size: " << bufferSize << std::endl;
        if (!pSpec || !pBuffer) {
            std::cerr << "Error allocating memory for filter." << std::endl;
            ippFree(pSpec);
            ippFree(pBuffer);
            return cv::Mat(); // Return empty image
        }
        std::cout << "Memory allocated successfully." << std::endl;

        // Initialize the filter
        status = ippiFilterBilateralBorderInit(
            ippiFilterBilateralGauss, // Filter type
            roiSize,
            kernelSize / 2, // Radius of circular neighborhood
            dataType,
            numChannels,
            distMethod,
            75.0f, // valSquareSigma
            75.0f, // posSquareSigma
            pSpec
        );

        std::cout << "Filter initialization status: " << status << std::endl;
        if (status != ippStsNoErr) {
            std::cerr << "Error initializing bilateral filter: " << status << std::endl;
            ippFree(pSpec);
            ippFree(pBuffer);
            return cv::Mat(); // Return empty image
        }
        std::cout << "Filter initialized successfully." << std::endl;

        // Allocate buffer for output image
        cv::Mat outputImage(inputImage.size(), inputImage.type());
        std::cout << "Output image size: " << outputImage.cols << " x " << outputImage.rows << ", type: " << outputImage.type() << std::endl;
        if (outputImage.empty()) {
            std::cerr << "Error allocating output image." << std::endl;
            ippFree(pSpec);
            ippFree(pBuffer);
            return cv::Mat(); // Return empty image
        }
        std::cout << "Output image allocated successfully." << std::endl;

        // Grayscale image
        status = ippiFilterBilateralBorder_8u_C1R(
            inputImage.ptr<Ipp8u>(),
            inputImage.step[0], // Row step (stride) in bytes
            outputImage.ptr<Ipp8u>(),
            outputImage.step[0], // Row step (stride) in bytes
            roiSize,
            ippBorderRepl,
            NULL,
            pSpec,
            pBuffer
        );
        std::cout << "Bilateral filter (grayscale) status: " << status << std::endl;
        if (status != ippStsNoErr) {
            std::cerr << "Error in filtering process." << std::endl;
        }
        else {
            std::cout << "Bilateral filter applied successfully." << std::endl;
        }

        // Free memory
        ippFree(pSpec);
        ippFree(pBuffer);

        return outputImage;
    }
    else if (numChannels == 3) {
        dataType = ipp8u; // BGR color
        distMethod = ippDistNormL2;

        // Calculate buffer size needed for the filter
        IppStatus status = ippiFilterBilateralGetBufferSize(ippiFilterBilateralGauss, roiSize, kernelSize, dataType, numChannels, distMethod, &specSize, &bufferSize);
        if (status != ippStsNoErr) {
            std::cerr << "Error getting buffer size: " << status << std::endl;
            return cv::Mat(); // Return empty image
        }

        // Allocate memory
        IppiFilterBilateralSpec* pSpec = (IppiFilterBilateralSpec*)ippMalloc(specSize);
        Ipp8u* pBuffer = (Ipp8u*)ippMalloc(bufferSize);
        if (!pSpec || !pBuffer) {
            std::cerr << "Error allocating memory for filter." << std::endl;
            ippFree(pSpec);
            ippFree(pBuffer);
            return cv::Mat(); // Return empty image
        }

        // Initialize the filter
        status = ippiFilterBilateralInit(ippiFilterBilateralGauss, roiSize, kernelSize, dataType, numChannels, distMethod, 75, 75, pSpec);
        if (status != ippStsNoErr) {
            std::cerr << "Error initializing bilateral filter: " << status << std::endl;
            ippFree(pSpec);
            ippFree(pBuffer);
            return cv::Mat(); // Return empty image
        }

        // Allocate buffer for output image
        cv::Mat outputImage(inputImage.size(), inputImage.type());

        status = ippiFilterBilateral_8u_C3R(inputImage.ptr<Ipp8u>(), inputImage.step, outputImage.ptr<Ipp8u>(), outputImage.step, roiSize, ippBorderRepl, NULL, pSpec, pBuffer);

        if (status != ippStsNoErr) {
            std::cerr << "Error applying bilateral filter: " << status << std::endl;
        }

        // Free memory
        ippFree(pSpec);
        ippFree(pBuffer);

        return outputImage;
    }
    else {
        std::cerr << "Unsupported number of channels: " << numChannels << std::endl;
        return cv::Mat(); // Return empty image
    }
    
}

//�̹����� ��踦 �����ϱ����� �ַ� ����
cv::Mat ImageProcessorIPP::sobelFilter(cv::Mat& inputImage)
{
    cv::Mat grayImage;
    if (inputImage.channels() == 1)
        grayImage = inputImage.clone();
    else
        grayImage = grayScale(inputImage);

    cv::Mat outputImage = cv::Mat::zeros(grayImage.size(), CV_16SC1);

    // IPP ���� ����
    IppiSize roiSize = { grayImage.cols, grayImage.rows };
    IppiMaskSize mask = ippMskSize3x3; // 3x3 �Һ� Ŀ�� ���
    IppNormType normType = ippNormL1;
    int bufferSize = 0;
    IppStatus status;

    // ���� ũ�� ���
    status = ippiFilterSobelGetBufferSize(roiSize, mask, normType, ipp8u, ipp16s, 1, &bufferSize);
    if (status != ippStsNoErr) {
        std::cerr << "���� ũ�� ��� ����: " << status << std::endl;
        return cv::Mat();
    }

    // ���� �Ҵ�
    Ipp8u* pBuffer = ippsMalloc_8u(bufferSize);
    if (!pBuffer) {
        std::cerr << "���� �Ҵ� ����." << std::endl;
        return cv::Mat();
    }

    // �Һ� ���� ����
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

    // ���� ����
    ippsFree(pBuffer);

    if (status != ippStsNoErr) {
        std::cerr << "�Һ� ���� ���� ����: " << status << std::endl;
        return cv::Mat();
    }

    // ���밪���� ��ȯ�ϰ� 8��Ʈ�� �����ϸ��Ͽ� �ð�ȭ
    cv::convertScaleAbs(outputImage, outputImage);

    // �÷� �̹����� ���, ����� ���� �÷� �̹��� ���� ��������
    if (inputImage.channels() == 3) {
        std::vector<cv::Mat> channels(3);
        cv::split(inputImage, channels);

        // �� ä�ο� �Һ� ��� �������� (���� �÷� �̹����� �ռ�)
        for (auto& channel : channels) {
            cv::addWeighted(channel, 0.5, outputImage, 0.5, 0, channel);
        }

        cv::merge(channels, outputImage);
    }

    return outputImage;
}