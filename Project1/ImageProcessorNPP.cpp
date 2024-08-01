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

    // ȸ�� ���� ����
    double angleInDegrees = isRight ? 270.0 : 90.0;
    double angleInRadians = angleInDegrees * CV_PI / 180.0; // ������ ���� ������ ��ȯ

    double cosAngle = std::abs(std::cos(angleInRadians));
    double sinAngle = std::abs(std::sin(angleInRadians));
    int dstWidth = static_cast<int>(srcWidth * cosAngle + srcHeight * sinAngle);
    int dstHeight = static_cast<int>(srcWidth * sinAngle + srcHeight * cosAngle);

    std::cout << "Cos(angle): " << cosAngle << std::endl;
    std::cout << "Sin(angle): " << sinAngle << std::endl;
    std::cout << "Calculated destination width: " << dstWidth << std::endl;
    std::cout << "Calculated destination height: " << dstHeight << std::endl;

    // ȸ���� �̹����� ���� Mat ���� (������ ����)
    cv::Mat outputImage(dstHeight, dstWidth, inputImage.type(), cv::Scalar(0));

    std::cout << "Output image size: " << outputImage.cols << " x " << outputImage.rows << std::endl;

    // GPU �޸𸮷� �̹��� ���ε�
    cv::cuda::GpuMat d_inputImage, d_outputImage;
    d_inputImage.upload(inputImage);
    d_outputImage.create(dstHeight, dstWidth, inputImage.type());

    std::cout << "GPU memory allocated for input and output images." << std::endl;
    std::cout << "Input image uploaded to GPU." << std::endl;
    std::cout << "Output image GPU Mat created with size: " << d_outputImage.cols << " x " << d_outputImage.rows << std::endl;

    // CUDA ��Ʈ�� ����
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::cout << "CUDA stream created." << std::endl;

    // NPP �Լ� ȣ���� ���� ������
    Npp8u* pSrc = d_inputImage.ptr<Npp8u>();
    Npp8u* pDst = d_outputImage.ptr<Npp8u>();

    std::cout << "Source GPU memory pointer: " << static_cast<void*>(pSrc) << std::endl;
    std::cout << "Destination GPU memory pointer: " << static_cast<void*>(pDst) << std::endl;

    // �̹��� ũ�� �� ��Ʈ���̵�
    NppiSize oSrcSize = { srcWidth, srcHeight };
    NppiSize oDstSize = { dstWidth, dstHeight };
    int srcStep = d_inputImage.step1();  // �Է� �̹����� �� ����
    int dstStep = d_outputImage.step1(); // ��� �̹����� �� ����

    std::cout << "Source image size: " << oSrcSize.width << " x " << oSrcSize.height << std::endl;
    std::cout << "Destination image size: " << oDstSize.width << " x " << oDstSize.height << std::endl;
    std::cout << "Source step: " << srcStep << std::endl;
    std::cout << "Destination step: " << dstStep << std::endl;

    // ���� �̹��� �߽� ���
    double srcCenterX = (srcWidth - 1) / 2.0;
    double srcCenterY = (srcHeight - 1) / 2.0;

    // ��� �̹��� �߽� ���
    double dstCenterX = (dstWidth - 1) / 2.0;
    double dstCenterY = (dstHeight - 1) / 2.0;

    std::cout << "Source center: (" << srcCenterX << ", " << srcCenterY << ")" << std::endl;
    std::cout << "Destination center: (" << dstCenterX << ", " << dstCenterY << ")" << std::endl;

    // ȸ�� �߽� ����
    double adjustedCenterX = dstCenterX;
    double adjustedCenterY = dstCenterY;

    // ��� �̹����� ROI�� ����
    NppiRect oSrcROI = { 0, 0, srcWidth, srcHeight };
    NppiRect oDstROI = { 0, 0, dstWidth, dstHeight };

    std::cout << "Source ROI: x=" << oSrcROI.x << ", y=" << oSrcROI.y
        << ", width=" << oSrcROI.width << ", height=" << oSrcROI.height << std::endl;
    std::cout << "Destination ROI: x=" << oDstROI.x << ", y=" << oDstROI.y
        << ", width=" << oDstROI.width << ", height=" << oDstROI.height << std::endl;

    NppStatus nppStatus;
    if (numChannels == 3) {
        // �÷� �̹��� ȸ��
        nppStatus = nppiRotate_8u_C3R(
            pSrc, oSrcSize, srcStep, oSrcROI,
            pDst, dstStep, oDstROI,
            angleInDegrees, adjustedCenterX, adjustedCenterY, NPPI_INTER_LINEAR
        );
    }
    else if (numChannels == 1) {
        // �׷��̽����� �̹��� ȸ��
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

    // NPP ���� ó��
    if (nppStatus != NPP_SUCCESS) {
        std::cerr << "NPP Error: " << nppStatus << std::endl;
        cudaStreamDestroy(stream);
        return cv::Mat();
    }

    std::cout << "NPP function executed successfully." << std::endl;

    // GPU �̹����� CPU �޸𸮷� �ٿ�ε�
    d_outputImage.download(outputImage);

    // �����: ��� �̹����� ���� ���� �ڳ� �ȼ� ���� Ȯ��
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

    // CUDA ��Ʈ�� �ı�
    cudaStreamDestroy(stream);
    std::cout << "CUDA stream destroyed." << std::endl;

    return outputImage;
   
    //cv::warpAffine ���
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

    // ȸ�� ���� ����
    double angleInDegrees = isRight ? 270.0 : 90.0;
    double angleInRadians = angleInDegrees * CV_PI / 180.0; // ������ ���� ������ ��ȯ

    double cosAngle = std::abs(std::cos(angleInRadians));
    double sinAngle = std::abs(std::sin(angleInRadians));
    int dstWidth = static_cast<int>(srcWidth * cosAngle + srcHeight * sinAngle);
    int dstHeight = static_cast<int>(srcWidth * sinAngle + srcHeight * cosAngle);

    std::cout << "Cos(angle): " << cosAngle << std::endl;
    std::cout << "Sin(angle): " << sinAngle << std::endl;
    std::cout << "Calculated destination width: " << dstWidth << std::endl;
    std::cout << "Calculated destination height: " << dstHeight << std::endl;

    // ȸ���� �̹����� ���� Mat ���� (������ ����)
    cv::Mat outputImage(dstHeight, dstWidth, inputImage.type(), cv::Scalar(0));

    std::cout << "Output image size: " << outputImage.cols << " x " << outputImage.rows << std::endl;

    // ȸ�� ��� ���
    cv::Point2f center(srcWidth / 2.0, srcHeight / 2.0);
    cv::Mat rotMat = cv::getRotationMatrix2D(center, angleInDegrees, 1.0);

    // ȸ�� ����� �����Ͽ� ��� �̹����� �߾��� �Է� �̹����� �߾Ӱ� �µ��� ��
    rotMat.at<double>(0, 2) += (dstWidth / 2.0) - center.x;
    rotMat.at<double>(1, 2) += (dstHeight / 2.0) - center.y;

    // �̹��� ȸ��
    cv::warpAffine(inputImage, outputImage, rotMat, outputImage.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

    // �����: ��� �̹����� ���� ���� �ڳ� �ȼ� ���� Ȯ��
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

    cv::Mat grayImage, blurredImage, edgeImage;
    cv::Mat outputImage;

    // Convert to grayscale if necessary
    if (inputImage.channels() == 3) {
        grayImage = grayScale(inputImage);
    }
    else {
        grayImage = inputImage.clone();
    }

    // Prepare GPU matrices
    cv::cuda::GpuMat d_src, d_sobelX, d_sobelY;
    d_src.upload(grayImage);

    d_sobelX.create(grayImage.size(), CV_16SC1);
    d_sobelY.create(grayImage.size(), CV_16SC1);

    // Define Sobel filter parameters
    NppiSize oSizeROI = { grayImage.cols, grayImage.rows };
    int srcStep = static_cast<int>(d_src.step);
    int dstStep = static_cast<int>(d_sobelX.step);

    // Allocate buffer for NPP functions
    int bufferSize = 0;
    NppStatus status;

    // Sobel X direction
    status = nppiFilterSobelHorizBorder_8u16s_C1R(
        d_src.ptr<Npp8u>(), srcStep, oSizeROI, { 0, 0 },
        d_sobelX.ptr<Npp16s>(), dstStep, oSizeROI,
        NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE
    );
    checkNPPStatus(status, "nppiFilterSobelHorizBorder_8u16s_C1R");

    // Sobel Y direction
    status = nppiFilterSobelVertBorder_8u16s_C1R(
        d_src.ptr<Npp8u>(), srcStep, oSizeROI, { 0, 0 },
        d_sobelY.ptr<Npp16s>(), dstStep, oSizeROI,
        NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE
    );
    checkNPPStatus(status, "nppiFilterSobelVertBorder_8u16s_C1R");

    // Download results from GPU
    cv::Mat sobelX, sobelY;
    d_sobelX.download(sobelX);
    d_sobelY.download(sobelY);

    // Calculate magnitude of gradient
    cv::Mat magnitude;
    cv::Mat sobelX_32f, sobelY_32f;
    sobelX.convertTo(sobelX_32f, CV_32F);
    sobelY.convertTo(sobelY_32f, CV_32F);
    cv::magnitude(sobelX_32f, sobelY_32f, magnitude);

    // Non-maximum suppression and hysteresis thresholding
    cv::Mat nonMaxSuppressed = cv::Mat::zeros(magnitude.size(), CV_8UC1);
    double lowThreshold = 50.0;
    double highThreshold = 150.0;

    for (int y = 1; y < magnitude.rows - 1; ++y) {
        for (int x = 1; x < magnitude.cols - 1; ++x) {
            float angle = atan2(sobelY_32f.at<float>(y, x), sobelX_32f.at<float>(y, x)) * 180.0 / CV_PI;
            angle = angle < 0 ? angle + 180 : angle;
            uchar& pixel = nonMaxSuppressed.at<uchar>(y, x);
            float currentMagnitude = magnitude.at<float>(y, x);

            // Non-maximum suppression
            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
                if (currentMagnitude >= magnitude.at<float>(y, x - 1) && currentMagnitude >= magnitude.at<float>(y, x + 1))
                    pixel = (currentMagnitude > highThreshold) ? 255 : (currentMagnitude > lowThreshold ? 128 : 0);
            }
            else if (angle >= 22.5 && angle < 67.5) {
                if (currentMagnitude >= magnitude.at<float>(y - 1, x + 1) && currentMagnitude >= magnitude.at<float>(y + 1, x - 1))
                    pixel = (currentMagnitude > highThreshold) ? 255 : (currentMagnitude > lowThreshold ? 128 : 0);
            }
            else if (angle >= 67.5 && angle < 112.5) {
                if (currentMagnitude >= magnitude.at<float>(y - 1, x) && currentMagnitude >= magnitude.at<float>(y + 1, x))
                    pixel = (currentMagnitude > highThreshold) ? 255 : (currentMagnitude > lowThreshold ? 128 : 0);
            }
            else {
                if (currentMagnitude >= magnitude.at<float>(y - 1, x - 1) && currentMagnitude >= magnitude.at<float>(y + 1, x + 1))
                    pixel = (currentMagnitude > highThreshold) ? 255 : (currentMagnitude > lowThreshold ? 128 : 0);
            }
        }
    }

    // Hysteresis thresholding
    cv::Mat finalEdges = cv::Mat::zeros(magnitude.size(), CV_8UC1);
    for (int y = 1; y < nonMaxSuppressed.rows - 1; ++y) {
        for (int x = 1; x < nonMaxSuppressed.cols - 1; ++x) {
            if (nonMaxSuppressed.at<uchar>(y, x) == 255) {
                finalEdges.at<uchar>(y, x) = 255;
            }
            else if (nonMaxSuppressed.at<uchar>(y, x) == 128) {
                // Check surrounding pixels for strong edges
                if (nonMaxSuppressed.at<uchar>(y - 1, x - 1) == 255 || nonMaxSuppressed.at<uchar>(y - 1, x) == 255 ||
                    nonMaxSuppressed.at<uchar>(y - 1, x + 1) == 255 || nonMaxSuppressed.at<uchar>(y, x - 1) == 255 ||
                    nonMaxSuppressed.at<uchar>(y, x + 1) == 255 || nonMaxSuppressed.at<uchar>(y + 1, x - 1) == 255 ||
                    nonMaxSuppressed.at<uchar>(y + 1, x) == 255 || nonMaxSuppressed.at<uchar>(y + 1, x + 1) == 255) {
                    finalEdges.at<uchar>(y, x) = 255;
                }
            }
        }
    }

    if (inputImage.channels() == 3) {
        // Color image processing
        outputImage = inputImage.clone();
        for (int y = 0; y < finalEdges.rows; ++y) {
            for (int x = 0; x < finalEdges.cols; ++x) {
                if (finalEdges.at<uchar>(y, x) > 0) {
                    outputImage.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0); // Green
                }
            }
        }
    }
    else {
        // Grayscale image processing
        outputImage = finalEdges.clone();
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
    cv::Mat grayImage, outputImage;

    // �׷��̽����� �̹����� ��ȯ (�÷� �̹����� ���)
    if (numChannels == 3) {
        cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
    }
    else if (numChannels == 1) {
        grayImage = inputImage.clone();
    }
    else {
        std::cerr << "Unsupported number of channels: " << numChannels << std::endl;
        return cv::Mat(); // �� �̹��� ��ȯ
    }

    // NPP �Լ��� �´� Ÿ�԰� ������ ����
    outputImage = cv::Mat::zeros(grayImage.size(), CV_8UC1);

    Npp8u* d_src;
    Npp8u* d_dst;
    NppStatus status;

    // CUDA �޸� �Ҵ�
    cudaMalloc(&d_src, grayImage.rows * grayImage.step);
    cudaMalloc(&d_dst, grayImage.rows * grayImage.step);

    // �Է� �̹����� GPU �޸𸮷� ����
    cudaMemcpy(d_src, grayImage.data, grayImage.rows * grayImage.step, cudaMemcpyHostToDevice);

    // IPP �Һ� ���� ����
    try {
        // ���� �Һ� ����
        status = nppiFilterSobelHoriz_8u_C1R(d_src, grayImage.step, d_dst, grayImage.step,
            { grayImage.cols, grayImage.rows });
        if (status != NPP_SUCCESS) {
            std::cerr << "nppiFilterSobelHoriz_8u_C1R failed with status: " << status << std::endl;
            throw std::runtime_error("nppiFilterSobelHoriz_8u_C1R failed.");
        }

        // ���� �Һ� ����
        status = nppiFilterSobelVert_8u_C1R(d_dst, grayImage.step, d_dst, grayImage.step,
            { grayImage.cols, grayImage.rows });
        if (status != NPP_SUCCESS) {
            std::cerr << "nppiFilterSobelVert_8u_C1R failed with status: " << status << std::endl;
            throw std::runtime_error("nppiFilterSobelVert_8u_C1R failed.");
        }

        // ����� ȣ��Ʈ�� ����
        cudaMemcpy(outputImage.data, d_dst, grayImage.rows * grayImage.step, cudaMemcpyDeviceToHost);
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

    // ���밪���� ��ȯ�ϰ� 8��Ʈ�� �����ϸ��Ͽ� �ð�ȭ
    cv::convertScaleAbs(outputImage, outputImage);

    // �÷� �̹����� ���, ����� ���� �÷� �̹��� ���� ��������
    if (numChannels == 3) {
        std::vector<cv::Mat> channels(3);
        cv::split(inputImage, channels);

        // �� ä�ο� �Һ� ��� �������� (���� �÷� �̹����� �ռ�)
        for (auto& channel : channels) {
            cv::Mat resizedOutput;
            if (outputImage.size() != channel.size()) {
                cv::resize(outputImage, resizedOutput, channel.size());
            }
            else {
                resizedOutput = outputImage;
            }
            cv::addWeighted(channel, 0.5, resizedOutput, 0.5, 0, channel);
        }

        cv::merge(channels, outputImage);
    }

    return outputImage;
}

cv::Mat ImageProcessorNPP::laplacianFilter(cv::Mat& inputImage)
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

    int srcWidth = inputImage.cols;
    int srcHeight = inputImage.rows;

    // GPU �޸𸮷� �̹��� ���ε�
    cv::cuda::GpuMat d_inputImage, d_outputImage;
    d_inputImage.upload(inputImage);

    // ���ö�þ� ���� Ŀ�� ����
    Npp32s laplacianKernel3x3[9] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };

    NppiPoint oSrcOffset = { 0, 0 };
    NppiMaskSize eMaskSize = NPP_MASK_SIZE_5_X_5;
    NppiBorderType eBorderType = NPP_BORDER_REPLICATE; // �����Ǵ� ��� ó�� ���

    NppiSize oSrcSize = { srcWidth, srcHeight };
    NppiSize oSizeROI = { srcWidth, srcHeight };

    int srcStep = d_inputImage.step;
    int dstStep = srcStep; // Destination step�� source step�� ����

    NppStatus nppStatus;

    cv::Mat outputImage; // Output image declaration
    outputImage.create(srcHeight, srcWidth, inputImage.type()); // Create the output image matrix

    // �׷��̽����� �̹����� ���ö�þ� ���� ����
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
    
    d_outputImage.download(outputImage);

    if (numChannels == 3) {        
        cv::Mat coloredEdgeImage;
        cv::cvtColor(outputImage, coloredEdgeImage, cv::COLOR_GRAY2BGR);  
        cv::addWeighted(inputImage, 0.5, coloredEdgeImage, 0.5, 0, outputImage);
    }      

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