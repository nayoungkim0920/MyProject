#include "ImageProcessorNPP.h"

ImageProcessorNPP::ImageProcessorNPP()
{
}

ImageProcessorNPP::~ImageProcessorNPP()
{
}

cv::Mat ImageProcessorNPP::grayScale(cv::Mat& inputImage) {
    if (inputImage.channels() != 3) {
        std::cerr << "�Է� �̹����� �÷� �̹������� �մϴ�." << std::endl;
        return cv::Mat();
    }

    // CUDA ��Ʈ�� ����
    cudaStream_t cudaStream;

    cudaStreamCreate(&cudaStream);

    // CUDA �޸� �Ҵ�
    Npp8u* d_src = nullptr;
    Npp8u* d_dst = nullptr;

    cudaMalloc(&d_src, inputImage.total() * inputImage.elemSize());
    cudaMalloc(&d_dst, inputImage.total() * sizeof(Npp8u)); // �׷��̽����� �̹���: 1 ä��

    cudaMemcpy(d_src, inputImage.data, inputImage.total() * inputImage.elemSize(), cudaMemcpyHostToDevice);

    // NPP ��� ���� (RGB�� �׷��̽����Ϸ� ��ȯ�ϱ� ���� �⺻ ���)
    Npp32f aCoeffs[3] = { 0.299f, 0.587f, 0.114f };

    // NPP ��Ʈ�� ���ؽ�Ʈ ���� �� �ʱ�ȭ
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = cudaStream;
    // nppStream �ʵ�� �ֽ� NPP ���̺귯�������� ������ ����

    // NPP �׷��̽����� ��ȯ �Լ� ȣ��
    NppiSize oSizeROI = { inputImage.cols, inputImage.rows };
    NppStatus status = nppiColorToGray_8u_C3C1R_Ctx(d_src, inputImage.cols * inputImage.elemSize(),
        d_dst, inputImage.cols,
        oSizeROI, aCoeffs, nppStreamCtx);

    if (status != NPP_SUCCESS) {
        std::cerr << "NPP �Լ� ȣ�� ����: " << status << std::endl;
        cudaFree(d_src);
        cudaFree(d_dst);
        cudaStreamDestroy(cudaStream);
        return cv::Mat();
    }

    // CUDA ��Ʈ�� ����ȭ
    cudaStreamSynchronize(cudaStream);

    // ����� ȣ��Ʈ �޸𸮷� ����
    cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC1);
    cudaMemcpy(outputImage.data, d_dst, outputImage.total() * outputImage.elemSize(), cudaMemcpyDeviceToHost);

    // CUDA �޸� ����
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaStreamDestroy(cudaStream);

    return outputImage;
}

cv::Mat ImageProcessorNPP::rotate(cv::Mat& inputImage, bool isRight) {
    if (inputImage.empty()) {
        std::cerr << "Input image is empty." << std::endl;
        return cv::Mat();
    }

    int srcWidth = inputImage.cols;
    int srcHeight = inputImage.rows;
    int numChannels = inputImage.channels();

    // ȸ�� ���� ����
    double angleInDegrees = isRight ? 90.0 : -90.0;
    double angleInRadians = angleInDegrees * CV_PI / 180.0; // ������ ���� ������ ��ȯ

    double cosAngle = std::abs(std::cos(angleInRadians));
    double sinAngle = std::abs(std::sin(angleInRadians));
    int dstWidth = static_cast<int>(srcWidth * cosAngle + srcHeight * sinAngle);
    int dstHeight = static_cast<int>(srcWidth * sinAngle + srcHeight * cosAngle);

    // ȸ���� �̹����� ���� Mat ����
    cv::Mat outputImage(dstHeight, dstWidth, inputImage.type());

    // GPU �޸𸮷� �̹��� ���ε�
    cv::cuda::GpuMat d_inputImage, d_outputImage;
    d_inputImage.upload(inputImage);
    d_outputImage.create(dstHeight, dstWidth, inputImage.type());

    // CUDA ��Ʈ�� ����
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // NPP �Լ� ȣ���� ���� ������
    Npp8u* pSrc = d_inputImage.ptr<Npp8u>();
    Npp8u* pDst = d_outputImage.ptr<Npp8u>();

    // �̹��� ũ�� �� ��Ʈ���̵�
    NppiSize oSrcSize = { srcWidth, srcHeight };
    NppiSize oDstSize = { dstWidth, dstHeight };
    int srcStep = d_inputImage.step1();  // �Է� �̹����� �� ����
    int dstStep = d_outputImage.step1(); // ��� �̹����� �� ����

    // ȸ�� �߽� ���� (�̹����� �߽�)
    double centerX = (srcWidth - 1) / 2.0;
    double centerY = (srcHeight - 1) / 2.0;

    // ��� �̹����� ROI�� ����
    NppiRect oSrcROI = { 0, 0, srcWidth, srcHeight };
    NppiRect oDstROI = { 0, 0, dstWidth, dstHeight };

    NppStatus nppStatus;
    if (numChannels == 3) {
        // �÷� �̹��� ȸ��
        nppStatus = nppiRotate_8u_C3R(
            pSrc, oSrcSize, srcStep, oSrcROI,
            pDst, dstStep, oDstROI,
            angleInDegrees, centerX, centerY, NPPI_INTER_LINEAR
        );
    }
    else if (numChannels == 1) {
        // �׷��̽����� �̹��� ȸ��
        nppStatus = nppiRotate_8u_C1R(
            pSrc, oSrcSize, srcStep, oSrcROI,
            pDst, dstStep, oDstROI,
            angleInDegrees, centerX, centerY, NPPI_INTER_LINEAR
        );
    }
    else {
        std::cerr << "Unsupported image format." << std::endl;
        return cv::Mat();
    }

    // NPP ���� ó��
    checkNPPError(nppStatus);

    // GPU �̹����� CPU �޸𸮷� �ٿ�ε�
    d_outputImage.download(outputImage);

    // CUDA ��Ʈ�� �ı�
    cudaStreamDestroy(stream);

    return outputImage;
}

cv::Mat ImageProcessorNPP::zoom(cv::Mat& inputImage, double newWidth, double newHeight) {
    
    int inputWidth = inputImage.cols;
    int inputHeight = inputImage.rows;
    int newWidthInt = static_cast<int>(newWidth);
    int newHeightInt = static_cast<int>(newHeight);

    // �����: �Է� �̹��� ũ�� ���
    std::cout << "Input Image Size: " << inputWidth << "x" << inputHeight << std::endl;
    std::cout << "New Image Size: " << newWidthInt << "x" << newHeightInt << std::endl;

    // GPU �޸𸮷� �̹��� ���ε�
    cv::cuda::GpuMat d_inputImage;
    d_inputImage.upload(inputImage);

    // GPU �޸𸮿� ��� �̹����� ���� GpuMat
    cv::cuda::GpuMat d_outputImage(newHeightInt, newWidthInt, inputImage.type());

    // NPP �Լ� ȣ���� ���� ������
    Npp8u* pSrc = d_inputImage.ptr<Npp8u>();
    Npp8u* pDst = d_outputImage.ptr<Npp8u>();

    // �̹��� ũ�� �� ��Ʈ���̵�
    NppiSize oSrcSize = { inputWidth, inputHeight };
    NppiSize oDstSize = { newWidthInt, newHeightInt };
    int srcStep = d_inputImage.step;  // �Է� �̹����� �� ����
    int dstStep = d_outputImage.step; // ��� �̹����� �� ����

    // �����: ��Ʈ���̵� �� ���
    std::cout << "Source Step: " << srcStep << std::endl;
    std::cout << "Destination Step: " << dstStep << std::endl;

    // �Է� �� ��� ROI ����
    NppiRect oSrcRectROI = { 0, 0, inputWidth, inputHeight };
    NppiRect oDstRectROI = { 0, 0, newWidthInt, newHeightInt };

    // �����: ROI �� ���
    std::cout << "Source ROI: " << oSrcRectROI.x << ", " << oSrcRectROI.y << ", " << oSrcRectROI.width << ", " << oSrcRectROI.height << std::endl;
    std::cout << "Destination ROI: " << oDstRectROI.x << ", " << oDstRectROI.y << ", " << oDstRectROI.width << ", " << oDstRectROI.height << std::endl;

    NppStatus nppStatus;

    if (inputImage.channels() == 3) {
        // NPP �÷� �̹��� �������� �Լ� ȣ��
        nppStatus = nppiResize_8u_C3R(
            pSrc, srcStep,         // �Է� �̹����� stride
            oSrcSize,              // �Է� �̹��� ũ��
            oSrcRectROI,           // �Է� �̹��� ����
            pDst, dstStep,         // ��� �̹����� stride
            oDstSize,              // ��� �̹��� ũ��
            oDstRectROI,           // ��� �̹��� ����
            NPPI_INTER_LINEAR      // ������ (���� ������)
        );
    }
    else if (inputImage.channels() == 1) {
        // NPP �׷��̽����� �̹��� �������� �Լ� ȣ��
        nppStatus = nppiResize_8u_C1R(
            pSrc, srcStep,         // �Է� �̹����� stride
            oSrcSize,              // �Է� �̹��� ũ��
            oSrcRectROI,           // �Է� �̹��� ����
            pDst, dstStep,         // ��� �̹����� stride
            oDstSize,              // ��� �̹��� ũ��
            oDstRectROI,           // ��� �̹��� ����
            NPPI_INTER_LINEAR      // ������ (���� ������)
        );
    }
    else {
        std::cerr << "Unsupported image format." << std::endl;
        return cv::Mat();
    }

    // NPP ���� ó��
    checkNPPError(nppStatus);

    // GPU �̹����� CPU �޸𸮷� �ٿ�ε�
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
        // �÷� �̹��� ó��
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
        // ��� �̹��� ó��
        outputImage = cv::Mat(grayImage.size(), CV_8UC1, cv::Scalar(0));
        for (int y = 0; y < edgeImage.rows; ++y) {
            for (int x = 0; x < edgeImage.cols; ++x) {
                if (edgeImage.at<uchar>(y, x) > 0) {
                    outputImage.at<uchar>(y, x) = 255; // ���
                }
            }
        }
    }

    return outputImage;
}

cv::Mat ImageProcessorNPP::medianFilter(cv::Mat& inputImage) {    

    if (inputImage.empty()) {
        std::cerr << "Input image is empty." << std::endl;
        return cv::Mat(); // �� �̹��� ��ȯ
    }

    int numChannels = inputImage.channels();
    cv::Mat outputImage(inputImage.size(), inputImage.type());

    Npp32s nSrcStep = inputImage.step[0];
    Npp32s nDstStep = outputImage.step[0];
    NppiSize oSizeROI = { inputImage.cols, inputImage.rows };
    NppiSize oMaskSize = { 5, 5 }; // 5x5 ����
    NppiPoint oAnchor = { 2, 2 }; // 5x5 ������ �߾�

    try {
        NppStatus status;
        Npp32u nBufferSize = 0;

        // CUDA �ʱ�ȭ
        cudaSetDevice(0);        

        // ���� �޸� �Ҵ� �� �̹��� ó��
        Npp8u* d_buffer;
        cudaMalloc(&d_buffer, nBufferSize);

        Npp8u* d_src;
        Npp8u* d_dst;
        size_t imageSize = inputImage.rows * inputImage.step;

        cudaMalloc(&d_src, imageSize);
        cudaMalloc(&d_dst, imageSize);

        // �Է� �̹����� GPU �޸𸮷� ����
        cudaMemcpy(d_src, inputImage.data, imageSize, cudaMemcpyHostToDevice);

        // �̵�� ���� ����
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

        // ����� ȣ��Ʈ�� ����
        cudaMemcpy(outputImage.data, d_dst, imageSize, cudaMemcpyDeviceToHost);

        // GPU �޸� ����
        cudaFree(d_src);
        cudaFree(d_dst);
        cudaFree(d_buffer);

        // CUDA ���� üũ
        cudaError_t cudaErr = cudaGetLastError();
        if (cudaErr != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(cudaErr) << std::endl;
            throw std::runtime_error("CUDA error occurred.");
        }

    }
    catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return cv::Mat(); // �� �̹��� ��ȯ
    }

    return outputImage;
}

cv::Mat ImageProcessorNPP::sobelFilter(cv::Mat& inputImage) {
    if (inputImage.empty()) {
        std::cerr << "Input image is empty." << std::endl;
        return cv::Mat(); // �� �̹��� ��ȯ
    }

    int numChannels = inputImage.channels();
    cv::Mat outputImage(inputImage.size(), inputImage.type());

    Npp32s nSrcStep = inputImage.step[0];
    Npp32s nDstStep = outputImage.step[0];
    NppiSize oSizeROI = { inputImage.cols, inputImage.rows };

    NppStatus status;

    // CUDA �ʱ�ȭ
    cudaSetDevice(0);

    size_t imageSize = inputImage.rows * inputImage.step;

    unsigned char* d_src;
    unsigned char* d_dst;

    // CUDA �޸� �Ҵ�
    cudaMalloc(&d_src, imageSize);
    cudaMalloc(&d_dst, imageSize);

    // �Է� �̹����� GPU �޸𸮷� ����
    cudaMemcpy(d_src, inputImage.data, imageSize, cudaMemcpyHostToDevice);

    // �Һ� ���� ó��
    try {
        if (numChannels == 3) {
            // ���� �Һ� ����
            status = nppiFilterSobelHoriz_8u_C3R(d_src, nSrcStep, d_dst, nDstStep, oSizeROI);
            if (status != NPP_SUCCESS) {
                std::cerr << "nppiFilterSobelHoriz_8u_C3R failed with status: " << status << std::endl;
                throw std::runtime_error("nppiFilterSobelHoriz_8u_C3R failed.");
            }

            // ���� �Һ� ����
            status = nppiFilterSobelVert_8u_C3R(d_dst, nSrcStep, d_dst, nDstStep, oSizeROI);
            if (status != NPP_SUCCESS) {
                std::cerr << "nppiFilterSobelVert_8u_C3R failed with status: " << status << std::endl;
                throw std::runtime_error("nppiFilterSobelVert_8u_C3R failed.");
            }
        }
        else if (numChannels == 1) {
            // �׷��̽����� �̹������� �Һ� ���� ����

            // ���� �Һ� ����
            status = nppiFilterSobelHoriz_8u_C1R(d_src, nSrcStep, d_dst, nDstStep, oSizeROI);
            if (status != NPP_SUCCESS) {
                std::cerr << "nppiFilterSobelHoriz_8u_C1R failed with status: " << status << std::endl;
                throw std::runtime_error("nppiFilterSobelHoriz_8u_C1R failed.");
            }

            // ���� �Һ� ����
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
            return cv::Mat(); // �� �̹��� ��ȯ
        }

        // ����� ȣ��Ʈ�� ����
        cudaMemcpy(outputImage.data, d_dst, imageSize, cudaMemcpyDeviceToHost);

        // CUDA ���� üũ
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
        return cv::Mat(); // �� �̹��� ��ȯ
    }

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