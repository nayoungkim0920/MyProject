//ImageProcessor.cpp
#include "ImageProcessor.h"

ImageProcessor::ImageProcessor(QObject* parent) : QObject(parent)
{
}

ImageProcessor::~ImageProcessor()
{
}

bool ImageProcessor::openImage(const std::string& fileName, cv::Mat& image)
{
    image = cv::imread(fileName);
    if (image.empty()) {
        qDebug() << "Failed to open image: " << QString::fromStdString(fileName);
        return false;
    }
    return true;
}

bool ImageProcessor::saveImage(const std::string& fileName, const cv::Mat& image)
{
    if (!cv::imwrite(fileName, image)) {
        qDebug() << "Failed to save image: " << QString::fromStdString(fileName);
        return false;
    }
    return true;
}

QFuture<bool> ImageProcessor::rotateImage(cv::Mat& imageOpenCV
                                        , cv::Mat& imageIPP
                                        , cv::Mat& imageCUDA
                                        , cv::Mat& imageCUDAKernel)
{
    //함수 이름을 문자열로 저장
    const char* functionName = __func__;

    return QtConcurrent::run([this
        , &imageOpenCV
        , &imageIPP
        , &imageCUDA
        , &imageCUDAKernel
        , functionName]() -> bool {

        QMutexLocker locker(&mutex);

        try {

            if (imageOpenCV.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            pushToUndoStackOpenCV(imageOpenCV.clone());
            pushToUndoStackIPP(imageIPP.clone());
            pushToUndoStackCUDA(imageCUDA.clone());
            pushToUndoStackCUDAKernel(imageCUDAKernel.clone());

            QVector<ProcessingResult> results;

            ProcessingResult outputOpenCV = rotateOpenCV(imageOpenCV);
            lastProcessedImageOpenCV = outputOpenCV.processedImage.clone();
            results.append(outputOpenCV);

            ProcessingResult outputIPP = rotateIPP(imageIPP);
            lastProcessedImageIPP = outputIPP.processedImage.clone();
            results.append(outputIPP);

            ProcessingResult outputCUDA = rotateCUDA(imageCUDA);
            lastProcessedImageCUDA = outputCUDA.processedImage.clone();
            results.append(outputCUDA);

            ProcessingResult outputCUDAKernel = rotateCUDAKernel(imageCUDAKernel);
            lastProcessedImageCUDAKernel = outputCUDAKernel.processedImage.clone();
            results.append(outputCUDAKernel);

            // 이미지 업데이트 및 시그널 발생
            emit imageProcessed(results);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "Exception occurred while rotating image:" << e.what();
            return false;
        }
        });
}

QFuture<bool> ImageProcessor::zoomOutImage(cv::Mat& imageOpenCV
                                            , cv::Mat& imageIPP
                                            , cv::Mat& imageCUDA
                                            , cv::Mat& imageCUDAKernel
                                            , double scaleFactor)
{
    //함수 이름을 문자열로 저장
    const char* functionName = __func__;

    return QtConcurrent::run([this
                            , &imageOpenCV
                            , &imageIPP
                            , &imageCUDA
                            , &imageCUDAKernel
                            , scaleFactor
                            , functionName]() -> bool {

            QMutexLocker locker(&mutex);

            try {

                if (imageOpenCV.empty()) {
                    qDebug() << "Input image is empty.";
                    return false;
                }

                if (scaleFactor <= 0) {
                    qDebug() << "Invalid scaling factor.";
                    return false;
                }

                pushToUndoStackOpenCV(imageOpenCV.clone());
                pushToUndoStackIPP(imageIPP.clone());
                pushToUndoStackCUDA(imageCUDA.clone());
                pushToUndoStackCUDAKernel(imageCUDAKernel.clone());

                QVector<ProcessingResult> results;

                double newWidth = static_cast<int>(imageOpenCV.cols * scaleFactor);
                double newHeight = static_cast<int>(imageOpenCV.rows * scaleFactor);

                ProcessingResult outputOpenCV = zoomOpenCV(imageOpenCV, newWidth, newHeight);
                lastProcessedImageOpenCV = outputOpenCV.processedImage.clone();
                results.append(outputOpenCV);

                ProcessingResult outputIPP = zoomIPP(imageIPP, newWidth, newHeight);
                lastProcessedImageIPP = outputIPP.processedImage.clone();
                results.append(outputIPP);

                ProcessingResult outputCUDA = zoomCUDA(imageCUDA, newWidth, newHeight);
                lastProcessedImageCUDA = outputCUDA.processedImage.clone();
                results.append(outputCUDA);

                ProcessingResult outputCUDAKernel = zoomCUDAKernel(imageCUDAKernel, newWidth, newHeight);
                lastProcessedImageCUDAKernel = outputCUDAKernel.processedImage.clone();
                results.append(outputCUDAKernel);

                emit imageProcessed(results); // 이미지 처리 완료 시그널 발생

                return true;
            }
            catch (const cv::Exception& e) {
                qDebug() << "이미지 축소 중 예외가 발생했습니다:" << e.what();
                return false;
            }
        });
}

QFuture<bool> ImageProcessor::zoomInImage(cv::Mat& imageOpenCV
                                        , cv::Mat& imageIPP
                                        , cv::Mat& imageCUDA
                                        , cv::Mat& imageCUDAKernel
                                        , double scaleFactor)
{
    //함수 이름을 문자열로 저장
    const char* functionName = __func__;

    return QtConcurrent::run([this
                            , &imageOpenCV
                            , &imageIPP
                            , &imageCUDA
                            , &imageCUDAKernel
                            , scaleFactor
                            , functionName]() -> bool {

        QMutexLocker locker(&mutex);

        try {

            if (imageOpenCV.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            if (scaleFactor <= 0) {
                qDebug() << "잘못된 확대 배율입니다.";
                return false;
            }

            pushToUndoStackOpenCV(imageOpenCV.clone());
            pushToUndoStackIPP(imageIPP.clone());
            pushToUndoStackCUDA(imageCUDA.clone());
            pushToUndoStackCUDAKernel(imageCUDAKernel.clone());

            QVector<ProcessingResult> results;

            double newWidth = static_cast<int>(imageOpenCV.cols * scaleFactor);
            double newHeight = static_cast<int>(imageOpenCV.rows * scaleFactor);

            ProcessingResult outputOpenCV = zoomOpenCV(imageOpenCV, newWidth, newHeight);
            lastProcessedImageOpenCV = outputOpenCV.processedImage.clone();
            results.append(outputOpenCV);

            ProcessingResult outputIPP = zoomIPP(imageIPP, newWidth, newHeight);
            lastProcessedImageIPP = outputIPP.processedImage.clone();
            results.append(outputIPP);

            ProcessingResult outputCUDA = zoomCUDA(imageCUDA, newWidth, newHeight);
            lastProcessedImageCUDA = outputCUDA.processedImage.clone();
            results.append(outputCUDA);

            ProcessingResult outputCUDAKernel = zoomCUDAKernel(imageCUDAKernel, newWidth, newHeight);
            lastProcessedImageCUDAKernel = outputCUDAKernel.processedImage.clone();
            results.append(outputCUDAKernel);

            emit imageProcessed(results); // 이미지 처리 완료 시그널 발생

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "이미지 확대 중 예외가 발생했습니다:" << e.what();
            return false;
        }
        });
}


// QDebug에서 cv::Size를 출력할 수 있도록 변환 함수 작성
QDebug operator<<(QDebug dbg, const cv::Size& size) {
    dbg.nospace() << "Size(width=" << size.width << ", height=" << size.height << ")";
    return dbg.space();
}

// QDebug에서 cv::Mat의 타입을 출력할 수 있도록 변환 함수 작성
QDebug operator<<(QDebug dbg, const cv::Mat& mat) {
    dbg.nospace() << "Mat(type=" << mat.type() << ", size=" << mat.size() << ")";
    return dbg.space();
}

QFuture<bool> ImageProcessor::grayScale(cv::Mat& imageOpenCV
                                        , cv::Mat& imageIPP
                                        , cv::Mat& imageCUDA
                                        , cv::Mat& imageCUDAKernel)
{
    const char* functionName = __func__;

    return QtConcurrent::run([this
                            , &imageOpenCV
                            , &imageIPP
                            , &imageCUDA
                            , &imageCUDAKernel
                            , functionName]() -> bool {
        
        QMutexLocker locker(&mutex);

        try {
            if (imageOpenCV.channels() != 3 && imageOpenCV.channels() != 1) {
                qDebug() << "Input image must be a 3-channel BGR image or already grayscale.";
                return false;
            }

            if (imageOpenCV.channels() == 3) {
                pushToUndoStackOpenCV(imageOpenCV.clone());
                pushToUndoStackIPP(imageIPP.clone());
                pushToUndoStackCUDA(imageCUDA.clone());
                pushToUndoStackCUDAKernel(imageCUDAKernel.clone());

                QVector<ProcessingResult> results;

                ProcessingResult outputOpenCV = grayScaleOpenCV(imageOpenCV);
                lastProcessedImageOpenCV = outputOpenCV.processedImage.clone();
                results.append(outputOpenCV);

                ProcessingResult outputIPP = grayScaleIPP(imageIPP);
                lastProcessedImageIPP = outputIPP.processedImage.clone();
                results.append(outputIPP);

                ProcessingResult outputCUDA = grayScaleCUDA(imageCUDA);
                lastProcessedImageCUDA = outputCUDA.processedImage.clone();
                results.append(outputCUDA);

                ProcessingResult outputCUDAKernel = grayScaleCUDAKernel(imageCUDAKernel);
                lastProcessedImageCUDAKernel = outputCUDAKernel.processedImage.clone();
                results.append(outputCUDAKernel);

                emit imageProcessed(results);
            }
            else {
                pushToUndoStackOpenCV(imageOpenCV.clone());
                pushToUndoStackIPP(imageIPP.clone());
                pushToUndoStackCUDA(imageCUDA.clone());
                pushToUndoStackCUDAKernel(imageCUDAKernel.clone());

                lastProcessedImageOpenCV = imageOpenCV.clone();
                lastProcessedImageIPP = imageIPP.clone();
                lastProcessedImageCUDA = imageCUDA.clone();
                lastProcessedImageCUDAKernel = imageCUDAKernel.clone();
            }

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "Exception occurred while converting to grayscale:" << e.what();
            return false;
        }
        });
}

/*bool ImageProcessor::grayScaleCUDA(cv::Mat& image)
{
    try {

        // CUDA 장치 설정
        cv::cuda::setDevice(0);

        // 입력 이미지를 CUDA GpuMat으로 업로드
        cv::cuda::GpuMat d_input;
        d_input.upload(image);

        // CUDA를 사용하여 그레이스케일로 변환
        cv::cuda::GpuMat d_output;
        cv::cuda::cvtColor(d_input, d_output, cv::COLOR_BGR2GRAY);

        // CUDA에서 호스트로 이미지 다운로드
        cv::Mat output;
        d_output.download(output);

        if (output.empty() || output.type() != CV_8UC1) {
            qDebug() << "Output image is empty or not in expected format after CUDA processing.";
            return false;
        }

        // 원본 이미지를 그레이스케일 이미지로 업데이트
        image = output.clone(); // 변환된 그레이스케일 이미지로 업데이트
        lastProcessedImage = image.clone(); // 마지막 처리된 이미지 업데이트

        return true;
    }
    catch (const cv::Exception& e) {
        qDebug() << "Exception occurred while converting to grayscale using CUDA:" << e.what();
        return false;
    }
}*/

ImageProcessor::ProcessingResult ImageProcessor::grayScaleOpenCV(cv::Mat& image)
{
    ProcessingResult result;
    result.functionName = "grayScale";
    result.processName = "OpenCV";

    double startTime = cv::getTickCount(); // 시작 시간 측정

    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // 원본 이미지를 그레이스케일 이미지로 업데이트
    //image = grayImage.clone(); // 변환된 그레이스케일 이미지로 업데이트
    //lastProcessedImage = image.clone(); // 마지막 처리된 이미지 업데이트

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result.processedImage = grayImage.clone();
    result.processingTime = elapsedTimeMs;

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::grayScaleIPP(cv::Mat& image)
{
    ProcessingResult result;
    result.functionName = "grayScale";
    result.processName = "IPP";

    double startTime = cv::getTickCount(); // 시작 시간 측정

    // IPP 사용을 위한 입력 및 출력 배열 설정
    IppiSize roiSize = { image.cols, image.rows };
    int srcStep = image.step;
    int dstStep = image.cols;
    Ipp8u* srcData = image.data;
    Ipp8u* dstData = ippsMalloc_8u(image.rows * image.cols);

    // IPP 그레이스케일 변환 (고정된 계수 사용)
    IppStatus status = ippiRGBToGray_8u_C3C1R(srcData, srcStep, dstData, dstStep, roiSize);

    if (status != ippStsNoErr) {
        std::cerr << "IPP 오류: " << status << std::endl;
        ippsFree(dstData); // 메모리 해제
        return result; // 오류 발생 시 처리 중단
    }

    // 결과를 OpenCV Mat으로 변환
    cv::Mat grayImage(image.rows, image.cols, CV_8UC1, dstData);

    // 원본 이미지를 그레이스케일 이미지로 업데이트
    //image = grayImage.clone(); // 변환된 그레이스케일 이미지로 업데이트
    //lastProcessedImage = image.clone(); // 마지막 처리된 이미지 업데이트

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result.processedImage = grayImage.clone();
    result.processingTime = elapsedTimeMs;

    ippsFree(dstData); // 메모리 해제

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::grayScaleCUDA(cv::Mat& image)
{
    ProcessingResult result;
    result.functionName = "grayScale";
    result.processName = "CUDA";

    double startTime = cv::getTickCount(); // 시작 시간 측정

    // CUDA 장치 설정
    //cv::cuda::setDevice(0);

    // 입력 이미지를 CUDA GpuMat으로 업로드
    cv::cuda::GpuMat d_input;
    d_input.upload(image);

    // CUDA를 사용하여 그레이스케일로 변환
    cv::cuda::GpuMat d_output;
    cv::cuda::cvtColor(d_input, d_output, cv::COLOR_BGR2GRAY);

    // CUDA에서 호스트로 이미지 다운로드
    cv::Mat grayImage;
    d_output.download(grayImage);

    if (grayImage.empty() || grayImage.type() != CV_8UC1) {
        qDebug() << "Output image is empty or not in expected format after CUDA processing.";
        return result;
    }

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result.processedImage = grayImage.clone();
    result.processingTime = elapsedTimeMs;

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::grayScaleCUDAKernel(cv::Mat& image)
{
    ProcessingResult result;
    result.functionName = "grayScale";
    result.processName = "CUDAKernel";

    double startTime = cv::getTickCount(); // 시작 시간 측정

    callGrayScaleImageCUDA(image);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result.processedImage = image.clone();
    result.processingTime = elapsedTimeMs;

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::zoomOpenCV(cv::Mat& image, double newWidth, double newHeight)
{
    ProcessingResult result;

    result.functionName = "zoomIn";
    result.processName = "OpenCV";

    double startTime = cv::getTickCount(); // 시작 시간 측정

    cv::Mat zoomInImage;
    cv::resize(image, zoomInImage, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result.processedImage = zoomInImage;
    result.processingTime = elapsedTimeMs;

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::zoomIPP(cv::Mat& image, double newWidth, double newHeight) {
    ProcessingResult result;
    result.functionName = "zoomIn";
    result.processName = "IPP";

    double startTime = cv::getTickCount(); // 시작 시간 측정

    cv::Mat zoomedImage;
    zoomedImage.create(static_cast<int>(newHeight), static_cast<int>(newWidth), image.type());

    Ipp8u* pSrcData = reinterpret_cast<Ipp8u*>(image.data);
    Ipp8u* pDstData = reinterpret_cast<Ipp8u*>(zoomedImage.data);

    IppiSize srcSize = { image.cols, image.rows };
    IppiSize dstSize = { static_cast<int>(newWidth), static_cast<int>(newHeight) };
    IppiPoint dstOffset = { 0, 0 };

    int specSize = 0;
    int initSize = 0;
    int bufSize = 0;
    std::vector<Ipp8u> pBuffer;
    IppiResizeSpec_32f* pSpec = nullptr;

    // Get the sizes of the spec and initialization buffers
    IppStatus status = ippiResizeGetSize_8u(srcSize, dstSize, ippNearest, 0, &specSize, &initSize);
    if (status != ippStsNoErr) {
        std::cerr << "Error: ippiResizeGetSize_8u failed with status code " << status << std::endl;
        return result;
    }

    // Allocate the spec and initialization buffers
    pSpec = (IppiResizeSpec_32f*)(ippMalloc(specSize));
    if (!pSpec) {
        std::cerr << "Error: Memory allocation failed for pSpec" << std::endl;
        return result;
    }

    pBuffer.resize(initSize);
    if (pBuffer.empty()) {
        std::cerr << "Error: Memory allocation failed for pBuffer" << std::endl;
        ippFree(pSpec);
        return result;
    }

    // Initialize the resize specification
    status = ippiResizeNearestInit_8u(srcSize, dstSize, pSpec);
    if (status != ippStsNoErr) {
        std::cerr << "Error: ippiResizeNearestInit_8u failed with status code " << status << std::endl;
        ippFree(pSpec);
        return result;
    }

    // Get the size of the working buffer
    status = ippiResizeGetBufferSize_8u(pSpec, dstSize, image.channels(), &bufSize);
    if (status != ippStsNoErr) {
        std::cerr << "Error: ippiResizeGetBufferSize_8u failed with status code " << status << std::endl;
        ippFree(pSpec);
        return result;
    }

    pBuffer.resize(bufSize);
    if (pBuffer.empty()) {
        std::cerr << "Error: Memory allocation failed for pBuffer" << std::endl;
        ippFree(pSpec);
        return result;
    }

    // Perform the resize operation
    if (image.channels() == 1) {
        status = ippiResizeNearest_8u_C1R(pSrcData, image.step[0], pDstData, zoomedImage.step[0], dstOffset, dstSize, pSpec, pBuffer.data());
    }
    else if (image.channels() == 3) {
        status = ippiResizeNearest_8u_C3R(pSrcData, image.step[0], pDstData, zoomedImage.step[0], dstOffset, dstSize, pSpec, pBuffer.data());
    }
    else {
        std::cerr << "Error: Unsupported number of channels" << std::endl;
        ippFree(pSpec);
        return result;
    }

    if (status != ippStsNoErr) {
        std::cerr << "Error: ippiResizeNearest_8u failed with status code " << status << std::endl;
        ippFree(pSpec);
        return result;
    }

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result.processedImage = zoomedImage.clone(); // 처리된 이미지 복사
    result.processingTime = elapsedTimeMs; // 처리 시간 설정

    // 메모리 해제
    ippFree(pSpec);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::zoomCUDA(cv::Mat& image, double newWidth, double newHeight)
{
    ProcessingResult result;
    result.functionName = "zoomIn";
    result.processName = "CUDA";

    double startTime = cv::getTickCount(); // 시작 시간 측정

    // GPU 메모리로 이미지 업로드
    cv::cuda::GpuMat d_image;
    d_image.upload(image);

    // 결과 이미지를 저장할 GPU 메모리 할당
    cv::cuda::GpuMat d_zoomInImage;

    // 이미지 크기 조정
    cv::cuda::resize(d_image, d_zoomInImage, cv::Size(static_cast<int>(newWidth), static_cast<int>(newHeight)), 0, 0, cv::INTER_LINEAR);

    // CPU 메모리로 결과 이미지 다운로드
    cv::Mat zoomedImage;
    d_zoomInImage.download(zoomedImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result.processedImage = zoomedImage.clone(); // 처리된 이미지 복사
    result.processingTime = elapsedTimeMs; // 처리 시간 설정

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::zoomCUDAKernel(cv::Mat& image, double newWidth, double newHeight)
{
    ProcessingResult result;
    result.functionName = "zoomIn";
    result.processName = "CUDAKernel";

    double startTime = cv::getTickCount(); // 시작 시간 측정

    callResizeImageCUDA(image, newWidth, newHeight);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result.processedImage = image.clone(); // 처리된 이미지 복사
    result.processingTime = elapsedTimeMs; // 처리 시간 설정

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::rotateOpenCV(cv::Mat& image)
{
    ProcessingResult result;
    result.functionName = "rotate";
    result.processName = "OpenCV";

    double startTime = cv::getTickCount(); // 시작 시간 측정

    cv::Mat rotatedImage;
    cv::rotate(image, rotatedImage, cv::ROTATE_90_CLOCKWISE);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result.processedImage = rotatedImage.clone(); // 처리된 이미지 복사
    result.processingTime = elapsedTimeMs; // 처리 시간 설정

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::rotateIPP(cv::Mat& image)
{
    ProcessingResult result;
    result.functionName = "rotate";
    result.processName = "IPP";

    double startTime = cv::getTickCount();

    // 입력 이미지의 크기
    IppiSize srcSize = { image.cols, image.rows };

    // 출력 이미지의 크기 설정: 오른쪽으로 회전된 이미지의 크기와 같도록 설정
    IppiSize dstSize = { image.rows, image.cols };

    // IPP에서 사용할 아핀 변환 계수
    double angle = 270.0;  // 90도 시계 방향으로 회전
    double xShift = static_cast<double>(srcSize.width);  // x 축 이동량: 이미지의 너비
    double yShift = 0.0;  // y 축 이동량: 0

    // 아핀 변환 계수 계산
    double coeffs[2][3];
    IppStatus status = ippiGetRotateTransform(angle, xShift, yShift, coeffs);
    if (status != ippStsNoErr) {
        std::cerr << "ippiGetRotateTransform error: " << status << std::endl;
        result.processedImage = image.clone();
        result.processingTime = 0.0;
        return result;
    }

    // IPP를 위한 필요한 변수들
    IppiWarpSpec* pSpec = nullptr;
    Ipp8u* pBuffer = nullptr;
    int specSize = 0, initSize = 0, bufSize = 0;
    const Ipp32u numChannels = 3;
    IppiBorderType borderType = ippBorderConst;
    IppiWarpDirection direction = ippWarpForward;
    Ipp64f pBorderValue[numChannels];
    for (int i = 0; i < numChannels; ++i) pBorderValue[i] = 255.0;

    // Spec 및 Init buffer 사이즈 설정
    status = ippiWarpAffineGetSize(srcSize, dstSize, ipp8u, coeffs, ippLinear, direction, borderType, &specSize, &initSize);
    if (status != ippStsNoErr) {
        std::cerr << "ippiWarpAffineGetSize error: " << status << std::endl;
        result.processedImage = image.clone();
        result.processingTime = 0.0;
        return result;
    }

    // Memory allocation
    pSpec = (IppiWarpSpec*)ippsMalloc_8u(specSize);
    if (pSpec == nullptr) {
        std::cerr << "Memory allocation error for pSpec" << std::endl;
        result.processedImage = image.clone();
        result.processingTime = 0.0;
        return result;
    }

    // Filter initialization
    status = ippiWarpAffineLinearInit(srcSize, dstSize, ipp8u, coeffs, direction, numChannels, borderType, pBorderValue, 0, pSpec);
    if (status != ippStsNoErr) {
        std::cerr << "ippiWarpAffineLinearInit error: " << status << std::endl;
        ippsFree(pSpec);
        result.processedImage = image.clone();
        result.processingTime = 0.0;
        return result;
    }

    // work buffer size
    status = ippiWarpGetBufferSize(pSpec, dstSize, &bufSize);
    if (status != ippStsNoErr) {
        std::cerr << "ippiWarpGetBufferSize error: " << status << std::endl;
        ippsFree(pSpec);
        result.processedImage = image.clone();
        result.processingTime = 0.0;
        return result;
    }

    pBuffer = ippsMalloc_8u(bufSize);
    if (pBuffer == nullptr) {
        std::cerr << "Memory allocation error for pBuffer" << std::endl;
        ippsFree(pSpec);
        result.processedImage = image.clone();
        result.processingTime = 0.0;
        return result;
    }

    // 회전된 이미지를 저장할 Mat 생성
    cv::Mat rotatedImage(dstSize.width, dstSize.height, image.type());

    // dstOffset 정의 (오른쪽으로 90도 회전 시)
    IppiPoint dstOffset = { 0, 0 };

    // IPP를 이용하여 이미지 회전
    status = ippiWarpAffineLinear_8u_C3R(image.data, srcSize.width * 3, rotatedImage.data, dstSize.width * 3, dstOffset, dstSize, pSpec, pBuffer);
    if (status != ippStsNoErr) {
        std::cerr << "ippiWarpAffineLinear_8u_C3R error: " << status << std::endl;
        ippsFree(pSpec);
        ippsFree(pBuffer);
        result.processedImage = image.clone();
        result.processingTime = 0.0;
        return result;
    }

    // 메모리 해제
    ippsFree(pSpec);
    ippsFree(pBuffer);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result.processedImage = rotatedImage.clone();
    result.processingTime = elapsedTimeMs; // 처리 시간 설정

    return result;
}



ImageProcessor::ProcessingResult ImageProcessor::rotateCUDA(cv::Mat& image)
{
    ProcessingResult result;
    result.functionName = "rotate";
    result.processName = "CUDA";

    double startTime = cv::getTickCount(); // 시작 시간 측정

    // #include <opencv2/cudawarping.hpp>
    double angle = 90.0; // 회전할 각도 (예: 90도)

    // 이미지를 GPU 메모리에 업로드
    cv::cuda::GpuMat gpuImage;
    gpuImage.upload(image);

    // 회전 중심을 이미지의 중앙으로 설정
    cv::Point2f center(gpuImage.cols / 2.0f, gpuImage.rows / 2.0f);

    // 회전 매트릭스 계산
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);

    // GPU에서 회전 수행
    cv::cuda::GpuMat gpuRotatedImage;
    cv::cuda::warpAffine(gpuImage, gpuRotatedImage, rotationMatrix, gpuImage.size());

    // 결과 이미지를 CPU 메모리로 다운로드
    cv::Mat rotatedImage;
    gpuRotatedImage.download(rotatedImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result.processedImage = rotatedImage.clone(); // 처리된 이미지 복사
    result.processingTime = elapsedTimeMs; // 처리 시간 설정

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::rotateCUDAKernel(cv::Mat& image)
{
    ProcessingResult result;
    result.functionName = "rotate";
    result.processName = "CUDAKernel";

    double startTime = cv::getTickCount(); // 시작 시간 측정

    callRotateImageCUDA(image);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result.processedImage = image.clone(); // 처리된 이미지 복사
    result.processingTime = elapsedTimeMs; // 처리 시간 설정

    return result;
}

double ImageProcessor::getCurrentTimeMs()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::steady_clock::now().time_since_epoch()).count();
}

QFuture<bool> ImageProcessor::gaussianBlur(cv::Mat& image, int kernelSize)
{
    //함수 이름을 문자열로 저장
    const char* functionName = __func__;

    return QtConcurrent::run([this, &image, kernelSize, functionName]() -> bool {

        QMutexLocker locker(&mutex);

        try {

            if (image.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            if (kernelSize % 2 == 0 || kernelSize < 1) {
                qDebug() << "Invalid kernel size for Gaussian blur.";
                return false;
            }

            //pushToUndoStack(image);

            // 처리시간계산 시작
            double startTime = getCurrentTimeMs();

            // Upload image to GPU
            //cv::cuda::GpuMat gpuImage;
            //gpuImage.upload(image);

            // Create Gaussian filter
            //cv::Ptr<cv::cuda::Filter> gaussianFilter =
            //    cv::cuda::createGaussianFilter(
            //        gpuImage.type(),
            //        gpuImage.type(),
            //        cv::Size(kernelSize, kernelSize),
            //        0);

            // Apply Gaussian blur on GPU
            //cv::cuda::GpuMat blurredGpuImage;
            //gaussianFilter->apply(gpuImage, blurredGpuImage);

            // Download the result back to CPU
            //cv::Mat blurredImage;
            //blurredGpuImage.download(blurredImage);

            //CUDA Kernel
            callGaussianBlurCUDA(image, kernelSize);

            // 처리시간계산 종료
            double endTime = getCurrentTimeMs();
            double processingTime = endTime - startTime;

            //image = blurredImage.clone();
            //lastProcessedImage = image.clone();

            //emit imageProcessed(image, processingTime, functionName);

            return true;

            /* OpenCV
            cv::Mat blurredImage;
            cv::GaussianBlur(image, blurredImage, cv::Size(kernelSize, kernelSize), 0, 0);

            image = blurredImage.clone();
            lastProcessedImage = image.clone();

            emit imageProcessed(image);

            return true;
            */
        }
        catch (const cv::Exception& e) {
            qDebug() << "Exception occurred while applying Gaussian blur:"
                << e.what();
            return false;
        }
        });
}

//Canny
QFuture<bool> ImageProcessor::cannyEdges(cv::Mat& image)
{
    //함수 이름을 문자열로 저장
    const char* functionName = __func__;

    return QtConcurrent::run([this, &image, functionName]() -> bool {
        QMutexLocker locker(&mutex);

        try {

            if (image.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            //pushToUndoStack(image);

            // 처리시간계산 시작
            double startTime = getCurrentTimeMs();

            //그레이스케일이 아닌경우
            if (image.channels() != 1)
            {
                //if (!grayScaleCUDA(image)) {
                //    return false;
                //}

                //CUDA Kernel
                callGrayScaleImageCUDA(image);
            }

            // GPU에서 캐니 엣지 감지기 생성
            //cv::cuda::GpuMat d_input(image);
            //cv::cuda::GpuMat d_cannyEdges;
            //cv::Ptr<cv::cuda::CannyEdgeDetector> cannyDetector = cv::cuda::createCannyEdgeDetector(50, 150);
            //cannyDetector->detect(d_input, d_cannyEdges);

            // 결과를 CPU 메모리로 복사
            //cv::Mat edges;
            //d_cannyEdges.download(edges);

            // 출력 이미지에 초록색 엣지 표시
            //cv::Mat outputImage = cv::Mat::zeros(image.size(), CV_8UC3); // 3-channel BGR image
            //cv::Mat mask(edges.size(), CV_8UC1, cv::Scalar(0)); // Mask for green edges
            //mask.setTo(cv::Scalar(255), edges); // Set pixels to 255 (white) where edges are detected
            //cv::Mat channels[3];
            //cv::split(outputImage, channels);
            //channels[1] = mask; // Green channel is set by mask
            //cv::merge(channels, 3, outputImage); // Merge channels to get green edges

            //CUDA Kernel
            callCannyEdgesCUDA(image);

            // 처리시간계산 종료
            double endTime = getCurrentTimeMs();
            double processingTime = endTime - startTime;

            //image = outputImage.clone();
            //lastProcessedImage = image.clone();

            // GPU 메모리 해제
            //d_cannyEdges.release();

            //emit imageProcessed(image, processingTime, functionName);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "Exception occurred while applying Canny edges:" << e.what();
            return false;
        }
        });
}

QFuture<bool> ImageProcessor::medianFilter(cv::Mat& image)
{

    //함수 이름을 문자열로 저장
    const char* functionName = __func__;

    return QtConcurrent::run([this, &image, functionName]()->bool {

        QMutexLocker locker(&mutex);

        try {

            if (image.empty()) {
                qDebug() << "median 필터를 적용할 이미지가 없습니다.";
                return false;
            }

            //pushToUndoStack(image);

            // 처리시간계산 시작
            double startTime = getCurrentTimeMs();

            // Upload image to GPU
            //cv::cuda::GpuMat gpuImage;
            //gpuImage.upload(image);

            // Create median filter
            //cv::Ptr<cv::cuda::Filter> medianFilter =
            //    cv::cuda::createMedianFilter(gpuImage.type(), 5);

            // Apply median filter on GPU
            //cv::cuda::GpuMat medianedGpuImage;
            //medianFilter->apply(gpuImage, medianedGpuImage);

            // Download the result back to CPU
            //cv::Mat medianedImage;
            //medianedGpuImage.download(medianedImage);

            //CUDA Kernel
            callMedianFilterCUDA(image);

            // 처리시간계산 종료
            double endTime = getCurrentTimeMs();
            double processingTime = endTime - startTime;

            //image = medianedImage.clone();
            //lastProcessedImage = image.clone();

            //emit imageProcessed(image, processingTime, functionName);

            return true;

            /*
            cv::Mat medianedImage;
            cv::medianBlur(image, medianedImage, 5);
            image = medianedImage.clone();
            lastProcessedImage = image.clone();

            emit imageProcessed(image);

            return true;
            */
        }
        catch (const cv::Exception& e) {
            qDebug() << "median 필터 적용 중 오류 발생: "
                << e.what();
            return false;
        }
        });
}

QFuture<bool> ImageProcessor::laplacianFilter(cv::Mat& image)
{
    //함수 이름을 문자열로 저장
    const char* functionName = __func__;

    return QtConcurrent::run([this, &image, functionName]()->bool {

        QMutexLocker locker(&mutex);

        try {

            if (image.empty()) {
                qDebug() << "laplacian 필터를 적용할 이미지가 없습니다.";
                return false;
            }

            //pushToUndoStack(image);

            // 처리시간계산 시작
            double startTime = getCurrentTimeMs();

            //cv::Mat filteredImage;
            //cv::Laplacian(image, filteredImage, CV_8U, 3);

            //CUDA Kernel
            callLaplacianFilterCUDA(image);

            // 처리시간계산 종료
            double endTime = getCurrentTimeMs();
            double processingTime = endTime - startTime;

            //image = filteredImage.clone();
            //lastProcessedImage = image.clone();

            //emit imageProcessed(image, processingTime, functionName);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "laplacian 필터 적용 중 오류 발생: "
                << e.what();
            return false;
        }
        });
}

QFuture<bool> ImageProcessor::bilateralFilter(cv::Mat& image)
{
    //함수 이름을 문자열로 저장
    const char* functionName = __func__;

    return QtConcurrent::run([this, &image, functionName]()->bool {

        QMutexLocker locker(&mutex);

        try {

            if (image.empty()) {
                qDebug() << "bilateral 필터를 적용할 이미지가 없습니다.";
                return false;
            }

            //pushToUndoStack(image);

            // 처리시간계산 시작
            double startTime = getCurrentTimeMs();

            //CUDA Kernel
            callBilateralFilterCUDA(image, 9, 75, 75);

            //cv::Mat filteredImage;
            //cv::bilateralFilter(image, filteredImage, 9, 75, 75);

            // 처리시간계산 종료
            double endTime = getCurrentTimeMs();
            double processingTime = endTime - startTime;

            //image = filteredImage.clone();
            //lastProcessedImage = image.clone();

            //emit imageProcessed(image, processingTime, functionName);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "bilateral 필터 적용 중 오류 발생: "
                << e.what();
            return false;
        }
        });
}

QFuture<bool> ImageProcessor::sobelFilter(cv::Mat& image)
{
    // 함수 이름을 문자열로 저장
    const char* functionName = __func__;

    return QtConcurrent::run([this, &image, functionName]()->bool {
        if (cv::cuda::getCudaEnabledDeviceCount() <= 0) {
            qDebug() << "No CUDA-enabled device found. Falling back to CPU implementation.";
            return false;
        }

        //pushToUndoStack(image);

        // 처리시간계산 시작
        double startTime = getCurrentTimeMs();

        //cv::cuda::GpuMat gpuImage, gpuGray, gpuSobelX, gpuSobelY;

        // 입력 이미지가 BGR 색상 포맷이 아닌 경우, BGR2GRAY 변환 수행
        //if (image.channels() != 3) {
        //    qDebug() << "Input image is not in BGR format. Converting to BGR...";
        //    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR); // 예시로 GRAY2BGR 사용. 실제로는 적절한 변환 사용
        //}

        //gpuImage.upload(image);
        //cv::cuda::cvtColor(gpuImage, gpuGray, cv::COLOR_BGR2GRAY);

        //cv::Ptr<cv::cuda::Filter> sobelX =
        //    cv::cuda::createSobelFilter(gpuGray.type(), CV_16S, 1, 0);
        //cv::Ptr<cv::cuda::Filter> sobelY =
        //    cv::cuda::createSobelFilter(gpuGray.type(), CV_16S, 0, 1);

        //sobelX->apply(gpuGray, gpuSobelX);
        //sobelY->apply(gpuGray, gpuSobelY);

        //cv::cuda::GpuMat sobelX_8U, sobelY_8U;
        //gpuSobelX.convertTo(sobelX_8U, CV_8U);
        //gpuSobelY.convertTo(sobelY_8U, CV_8U);

        //cv::cuda::addWeighted(sobelX_8U, 0.5, sobelY_8U, 0.5, 0, gpuGray);

        //cv::Mat sobeledImage;
        //gpuGray.download(sobeledImage);

        //CUDA Kernel
        callSobelFilterCUDA(image);

        // 처리시간계산 종료
        double endTime = getCurrentTimeMs();
        double processingTime = endTime - startTime;

        //image = sobeledImage.clone();
        //lastProcessedImage = image.clone();

        //emit imageProcessed(image, processingTime, functionName);

        return true;
        });
}


bool ImageProcessor::canUndoOpenCV() const
{
    return !undoStackOpenCV.empty();
}

bool ImageProcessor::canRedoOpenCV() const
{
    return !redoStackOpenCV.empty();
}

//실행취소
// Undo operation
void ImageProcessor::undo()
{
    const char* functionName = __func__;
    QVector<ProcessingResult> results;

    try {
        if (!canUndoOpenCV()) {
            throw std::runtime_error("Cannot undo: Undo stack is empty");
        }

        double startTime = cv::getTickCount(); // 시작 시간 측정        

        // 현재 이미지를 redo 스택에 푸시
        redoStackOpenCV.push(lastProcessedImageOpenCV.clone());
        redoStackIPP.push(lastProcessedImageIPP.clone());
        redoStackCUDA.push(lastProcessedImageCUDA.clone());
        redoStackCUDAKernel.push(lastProcessedImageCUDAKernel.clone());

        // undo 스택에서 이미지 복원
        lastProcessedImageOpenCV = undoStackOpenCV.top().clone();
        lastProcessedImageIPP = undoStackIPP.top().clone();
        lastProcessedImageCUDA = undoStackCUDA.top().clone();
        lastProcessedImageCUDAKernel = undoStackCUDAKernel.top().clone();

        // undo 스택에서 이미지 제거
        undoStackOpenCV.pop();
        undoStackIPP.pop();
        undoStackCUDA.pop();
        undoStackCUDAKernel.pop();

        double endTime = cv::getTickCount(); // 종료 시간 측정
        double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

        // 결과 생성
        results.append(ProcessingResult(functionName, "OpenCV", lastProcessedImageOpenCV.clone(), elapsedTimeMs));
        results.append(ProcessingResult(functionName, "IPP", lastProcessedImageIPP.clone(), elapsedTimeMs));
        results.append(ProcessingResult(functionName, "CUDA", lastProcessedImageCUDA.clone(), elapsedTimeMs));
        results.append(ProcessingResult(functionName, "CUDAKernel", lastProcessedImageCUDAKernel.clone(), elapsedTimeMs));

        emit imageProcessed(results);
    }
    catch (const std::exception& e) {
        qDebug() << "Exception occurred in ImageProcessor::undo(): " << e.what();
    }
}


//재실행
void ImageProcessor::redo()
{
    const char* functionName = __func__;
    QVector<ProcessingResult> results;

    try {
        if (!canRedoOpenCV()) {
            throw std::runtime_error("Cannot redo: Redo stack is empty");
        }

        double startTime = cv::getTickCount(); // 시작 시간 측정        

        // 현재 이미지를 undo 스택에 푸시
        undoStackOpenCV.push(lastProcessedImageOpenCV.clone());
        undoStackIPP.push(lastProcessedImageIPP.clone());
        undoStackCUDA.push(lastProcessedImageCUDA.clone());
        undoStackCUDAKernel.push(lastProcessedImageCUDAKernel.clone());

        // redo 스택에서 이미지 복원
        lastProcessedImageOpenCV = redoStackOpenCV.top().clone();
        lastProcessedImageIPP = redoStackIPP.top().clone();
        lastProcessedImageCUDA = redoStackCUDA.top().clone();
        lastProcessedImageCUDAKernel = redoStackCUDAKernel.top().clone();

        // redo 스택에서 이미지 제거
        redoStackOpenCV.pop();
        redoStackIPP.pop();
        redoStackCUDA.pop();
        redoStackCUDAKernel.pop();

        double endTime = cv::getTickCount(); // 종료 시간 측정
        double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

        // 결과 생성
        results.append(ProcessingResult(functionName, "OpenCV", lastProcessedImageOpenCV.clone(), elapsedTimeMs));
        results.append(ProcessingResult(functionName, "IPP", lastProcessedImageIPP.clone(), elapsedTimeMs));
        results.append(ProcessingResult(functionName, "CUDA", lastProcessedImageCUDA.clone(), elapsedTimeMs));
        results.append(ProcessingResult(functionName, "CUDAKernel", lastProcessedImageCUDAKernel.clone(), elapsedTimeMs));

        emit imageProcessed(results);
    }
    catch (const std::exception& e) {
        qDebug() << "Exception occurred in ImageProcessor::redo(): " << e.what();
    }
}



void ImageProcessor::cleanUndoStack()
{
    QMutexLocker locker(&mutex);
    while (!undoStackOpenCV.empty()) {
        undoStackOpenCV.pop();
    }

    while (!undoStackIPP.empty()) {
        undoStackIPP.pop();
    }

    while (!undoStackCUDA.empty()) {
        undoStackCUDA.pop();
    }

    while (!undoStackCUDAKernel.empty()) {
        undoStackCUDAKernel.pop();
    }
}

void ImageProcessor::cleanRedoStack()
{
    QMutexLocker locker(&mutex);
    while (!redoStackOpenCV.empty()) {
        redoStackOpenCV.pop();
    }

    while (!redoStackIPP.empty()) {
        redoStackIPP.pop();
    }

    while (!redoStackCUDA.empty()) {
        redoStackCUDA.pop();
    }

    while (!redoStackCUDAKernel.empty()) {
        redoStackCUDAKernel.pop();
    }
}

void ImageProcessor::initializeCUDA()
{
    // 임의의 작은 작업을 수행하여 CUDA 초기화를 유도
    cv::cuda::GpuMat temp;
    temp.upload(cv::Mat::zeros(1, 1, CV_8UC1));
    cv::cuda::cvtColor(temp, temp, cv::COLOR_GRAY2BGR);
}

const cv::Mat& ImageProcessor::getLastProcessedImageOpenCV() const
{
    return lastProcessedImageOpenCV;
}

const cv::Mat& ImageProcessor::getLastProcessedImageIPP() const
{
    return lastProcessedImageIPP;
}

const cv::Mat& ImageProcessor::getLastProcessedImageCUDA() const
{
    return lastProcessedImageCUDA;
}

const cv::Mat& ImageProcessor::getLastProcessedImageCUDAKernel() const
{
    return lastProcessedImageCUDAKernel;
}

void ImageProcessor::pushToUndoStackOpenCV(const cv::Mat& image)
{
    undoStackOpenCV.push(image.clone());
}

void ImageProcessor::pushToUndoStackIPP(const cv::Mat& image)
{
    undoStackIPP.push(image.clone());
}

void ImageProcessor::pushToUndoStackCUDA(const cv::Mat& image)
{
    undoStackCUDA.push(image.clone());
}

void ImageProcessor::pushToUndoStackCUDAKernel(const cv::Mat& image)
{
    undoStackCUDAKernel.push(image.clone());
}

void ImageProcessor::pushToRedoStackOpenCV(const cv::Mat& image)
{
    redoStackOpenCV.push(image.clone());
}

void ImageProcessor::pushToRedoStackIPP(const cv::Mat& image)
{
    redoStackIPP.push(image.clone());
}

void ImageProcessor::pushToRedoStackCUDA(const cv::Mat& image)
{
    redoStackCUDA.push(image.clone());
}

void ImageProcessor::pushToRedoStackCUDAKernel(const cv::Mat& image)
{
    redoStackCUDAKernel.push(image.clone());
}
