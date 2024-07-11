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

ImageProcessor::ProcessingResult ImageProcessor::grayScaleOpenCV(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    cv::Mat outputImage = convertToGrayOpenCV(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "grayScale", "OpenCV", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::grayScaleIPP(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    // 입력 이미지가 3채널 BGR 이미지인지 확인
    //if (inputImage.channels() != 3 || inputImage.type() != CV_8UC3) {
        //std::cerr << "Warning: Input image is not a 3-channel BGR image. Converting to BGR." << std::endl;
    //    cv::Mat temp;
    //    cv::cvtColor(inputImage, temp, cv::COLOR_GRAY2BGR);
    //    inputImage = temp;
    //}

    cv::Mat outputImage = ImageProcessor::convertToGrayIPP(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산   

    result = setResult(result, inputImage, outputImage, "grayScale", "IPP", elapsedTimeMs);

    return result;
}

cv::Mat ImageProcessor::convertToGrayOpenCV(const cv::Mat& inputImage)
{
    cv::Mat grayImage;
    cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);

    return grayImage;
}

cv::Mat ImageProcessor::convertToGrayIPP(const cv::Mat& inputImage)
{
    // IPP 초기화
    ippInit();

    // 입력 이미지의 크기 및 스텝 설정
    IppiSize roiSize = { inputImage.cols, inputImage.rows };
    int srcStep = inputImage.step;
    int dstStep = inputImage.cols;
    Ipp8u* srcData = inputImage.data;

    // 출력 이미지 생성 및 IPP 메모리 할당
    cv::Mat grayImage(inputImage.rows, inputImage.cols, CV_8UC1);
    Ipp8u* dstData = grayImage.data;

    // IPP RGB to Gray 변환 수행
    IppStatus status = ippiRGBToGray_8u_C3C1R(srcData, srcStep, dstData, dstStep, roiSize);
    if (status != ippStsNoErr) {
        std::cerr << "IPP 오류: " << status << std::endl;
        return cv::Mat(); // 오류 발생 시 빈 Mat 반환
    }

    return grayImage;
}

cv::Mat ImageProcessor::convertToGrayCUDA(const cv::Mat& inputImage)
{
    // 입력 이미지를 CUDA GpuMat으로 업로드
    cv::cuda::GpuMat d_input;
    d_input.upload(inputImage);

    // CUDA를 사용하여 그레이스케일로 변환
    cv::cuda::GpuMat d_output;
    cv::cuda::cvtColor(d_input, d_output, cv::COLOR_BGR2GRAY);

    // CUDA에서 호스트로 이미지 다운로드
    cv::Mat grayImage;
    d_output.download(grayImage);

    return grayImage;
}

cv::Mat ImageProcessor::convertToGrayCUDAKernel(cv::Mat& inputImage)
{
    cv::Mat grayImage;
    callGrayScaleImageCUDA(inputImage, grayImage);

    return grayImage;
}

ImageProcessor::ProcessingResult ImageProcessor::grayScaleCUDA(cv::Mat& inputImage)
{    
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    cv::Mat outputImage = convertToGrayCUDA(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "grayScale", "CUDA", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::grayScaleCUDAKernel(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    cv::Mat outputImage = convertToGrayCUDAKernel(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "grayScale", "CUDAKernel", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::zoomOpenCV(cv::Mat& inputImage, double newWidth, double newHeight)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    cv::Mat outputImage;
    cv::resize(inputImage, outputImage, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "zoom", "OpenCV", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::zoomIPP(cv::Mat& inputImage, double newWidth, double newHeight) {
 
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    // IPP 변수들 선언
    IppStatus status;
    IppiSize srcSize = { inputImage.cols, inputImage.rows };
    IppiSize dstSize = { static_cast<int>(newWidth), static_cast<int>(newHeight) };
    IppiPoint dstOffset = { 0, 0 };
    std::vector<Ipp8u> pBuffer;
    IppiResizeSpec_32f* pSpec = nullptr;

    // 크기 및 초기화 버퍼 할당
    int specSize = 0, initSize = 0, bufSize = 0;
    status = ippiResizeGetSize_8u(srcSize, dstSize, ippNearest, 0, &specSize, &initSize);
    if (status != ippStsNoErr) {
        std::cerr << "Error: ippiResizeGetSize_8u failed with status code " << status << std::endl;
        return result;
    }

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

    // 크기 조정 스펙 초기화
    status = ippiResizeNearestInit_8u(srcSize, dstSize, pSpec);
    if (status != ippStsNoErr) {
        std::cerr << "Error: ippiResizeNearestInit_8u failed with status code " << status << std::endl;
        ippFree(pSpec);
        return result;
    }

    // Get the size of the working buffer
    status = ippiResizeGetBufferSize_8u(pSpec, dstSize, inputImage.channels(), &bufSize);
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

    // 크기 조정 수행
    cv::Mat outputImage(dstSize.height, dstSize.width, inputImage.type());
    Ipp8u* pSrcData = reinterpret_cast<Ipp8u*>(inputImage.data);
    Ipp8u* pDstData = reinterpret_cast<Ipp8u*>(outputImage.data);

    // 이미지 타입에 따라 IPP 함수 호출
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
    else {
        std::cerr << "Error: Unsupported image type" << std::endl;
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

    result = setResult(result, inputImage, outputImage, "zoom", "IPP", elapsedTimeMs);

    // 메모리 해제
    ippFree(pSpec);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::zoomCUDA(cv::Mat& inputImage, double newWidth, double newHeight)
{    
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

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

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "zoom", "CUDA", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::zoomCUDAKernel(cv::Mat& inputImage, double newWidth, double newHeight)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    cv::Mat outputImage;
    callZoomImageCUDA(inputImage, outputImage, newWidth, newHeight);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "zoom", "CUDAKernel", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::rotateOpenCV(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    cv::Mat outputImage;
    cv::rotate(inputImage, outputImage, cv::ROTATE_90_CLOCKWISE);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "rotate", "OpenCV", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::rotateIPP(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount();

    // 입력 이미지의 크기
    IppiSize srcSize = { inputImage.cols, inputImage.rows };

    // 출력 이미지의 크기 설정: 오른쪽으로 회전된 이미지의 크기와 같도록 설정
    IppiSize dstSize = { inputImage.rows, inputImage.cols };

    // IPP에서 사용할 아핀 변환 계수
    double angle = 270.0;  // 90도 시계 방향으로 회전
    double xShift = static_cast<double>(srcSize.width);  // x 축 이동량: 이미지의 너비
    double yShift = 0.0;  // y 축 이동량: 0

    // 아핀 변환 계수 계산
    double coeffs[2][3];
    IppStatus status = ippiGetRotateTransform(angle, xShift, yShift, coeffs);
    if (status != ippStsNoErr) {
        std::cerr << "ippiGetRotateTransform error: " << status << std::endl;
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
        return result;
    }

    // Memory allocation
    pSpec = (IppiWarpSpec*)ippsMalloc_8u(specSize);
    if (pSpec == nullptr) {
        std::cerr << "Memory allocation error for pSpec" << std::endl;
        return result;
    }

    // Filter initialization
    status = ippiWarpAffineLinearInit(srcSize, dstSize, ipp8u, coeffs, direction, numChannels, borderType, pBorderValue, 0, pSpec);
    if (status != ippStsNoErr) {
        std::cerr << "ippiWarpAffineLinearInit error: " << status << std::endl;
        ippsFree(pSpec);
        return result;
    }

    // work buffer size
    status = ippiWarpGetBufferSize(pSpec, dstSize, &bufSize);
    if (status != ippStsNoErr) {
        std::cerr << "ippiWarpGetBufferSize error: " << status << std::endl;
        ippsFree(pSpec);
        return result;
    }

    pBuffer = ippsMalloc_8u(bufSize);
    if (pBuffer == nullptr) {
        std::cerr << "Memory allocation error for pBuffer" << std::endl;
        ippsFree(pSpec);
        return result;
    }

    // 회전된 이미지를 저장할 Mat 생성
    cv::Mat outputImage(dstSize.width, dstSize.height, inputImage.type());

    // dstOffset 정의 (오른쪽으로 90도 회전 시)
    IppiPoint dstOffset = { 0, 0 };

    // IPP를 이용하여 이미지 회전
    status = ippiWarpAffineLinear_8u_C3R(inputImage.data, srcSize.width * 3, outputImage.data, dstSize.width * 3, dstOffset, dstSize, pSpec, pBuffer);
    if (status != ippStsNoErr) {
        std::cerr << "ippiWarpAffineLinear_8u_C3R error: " << status << std::endl;
        ippsFree(pSpec);
        ippsFree(pBuffer);
        return result;
    }   

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "rotate", "IPP", elapsedTimeMs);

    // 메모리 해제
    ippsFree(pSpec);
    ippsFree(pBuffer);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::rotateCUDA(cv::Mat& inputImage)
{
    ProcessingResult result;

    double startTime = cv::getTickCount(); // 시작 시간 측정

    // #include <opencv2/cudawarping.hpp>
    double angle = 90.0; // 회전할 각도 (예: 90도)

    // 이미지를 GPU 메모리에 업로드
    cv::cuda::GpuMat gpuImage;
    gpuImage.upload(inputImage);

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

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "rotate", "CUDA", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::rotateCUDAKernel(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    cv::Mat outputImage;
    callRotateImageCUDA(inputImage, outputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "rotate", "CUDAKernel", elapsedTimeMs);

    return result;
}

double ImageProcessor::getCurrentTimeMs()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::steady_clock::now().time_since_epoch()).count();
}

QFuture<bool> ImageProcessor::gaussianBlur(cv::Mat& imageOpenCV
                                            , cv::Mat& imageIPP
                                            , cv::Mat& imageCUDA
                                            , cv::Mat& imageCUDAKernel
                                            , int kernelSize)
{
    //함수 이름을 문자열로 저장
    const char* functionName = __func__;

    return QtConcurrent::run([this
        , &imageOpenCV
        , &imageIPP
        , &imageCUDA
        , &imageCUDAKernel
        , kernelSize
        , functionName]() -> bool {

        QMutexLocker locker(&mutex);

        try {

            if (imageOpenCV.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            if (kernelSize % 2 == 0 || kernelSize < 1) {
                qDebug() << "Invalid kernel size for Gaussian blur.";
                return false;
            }

            pushToUndoStackOpenCV(imageOpenCV.clone());
            pushToUndoStackIPP(imageIPP.clone());
            pushToUndoStackCUDA(imageCUDA.clone());
            pushToUndoStackCUDAKernel(imageCUDAKernel.clone());

            QVector<ProcessingResult> results;

            ProcessingResult outputOpenCV = gaussianBlurOpenCV(imageOpenCV, kernelSize);
            lastProcessedImageOpenCV = outputOpenCV.processedImage.clone();
            results.append(outputOpenCV);

            ProcessingResult outputIPP = gaussianBlurIPP(imageIPP, kernelSize);
            lastProcessedImageIPP = outputIPP.processedImage.clone();
            results.append(outputIPP);

            ProcessingResult outputCUDA = gaussianBlurCUDA(imageCUDA, kernelSize);
            lastProcessedImageCUDA = outputCUDA.processedImage.clone();
            results.append(outputCUDA);

            ProcessingResult outputCUDAKernel = gaussianBlurCUDAKernel(imageCUDAKernel, kernelSize);
            lastProcessedImageCUDAKernel = outputCUDAKernel.processedImage.clone();
            results.append(outputCUDAKernel);

            emit imageProcessed(results);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "Exception occurred while applying Gaussian blur:"
                << e.what();
            return false;
        }
        });
}

ImageProcessor::ProcessingResult ImageProcessor::gaussianBlurOpenCV(cv::Mat& inputImage, int kernelSize)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    cv::Mat outputImage;
    cv::GaussianBlur(inputImage, outputImage, cv::Size(kernelSize, kernelSize), 0, 0);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "gaussianBlur", "OpenCV", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::gaussianBlurIPP(cv::Mat& inputImage, int kernelSize) {
    
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    // 입력 이미지가 3채널 BGR 이미지인지 확인하고, 아닌 경우 변환
    cv::Mat bgrImage;
    if (inputImage.channels() != 3 || inputImage.type() != CV_8UC3) {
        //std::cerr << "Warning: Input image is not a 3-channel BGR image. Converting to BGR." << std::endl;
        if (inputImage.channels() == 1) {
            cv::cvtColor(inputImage, bgrImage, cv::COLOR_GRAY2BGR);
        }
        else {
            std::cerr << "Error: Unsupported image format." << std::endl;
            return result;
        }
    }
    else {
        bgrImage = inputImage.clone(); // 이미 BGR인 경우 그대로 사용
    }

    // 출력 이미지를 16비트 3채널(CV_16UC3)로 선언
    cv::Mat outputImage(bgrImage.size(), CV_16UC3);

    // IPP 함수에 전달할 포인터들
    Ipp16u* pSrc = reinterpret_cast<Ipp16u*>(bgrImage.data);
    Ipp16u* pDst = reinterpret_cast<Ipp16u*>(outputImage.data);

    // ROI 크기 설정
    IppiSize roiSize = { bgrImage.cols, bgrImage.rows };

    // 필터링을 위한 버퍼 및 스펙트럼 크기 계산
    int specSize, bufferSize;
    IppStatus status = ippiFilterGaussianGetBufferSize(roiSize, kernelSize, ipp16u, 3, &specSize, &bufferSize);
    if (status != ippStsNoErr) {
        std::cerr << "Error: ippiFilterGaussianGetBufferSize failed with status " << status << std::endl;
        return result; // 빈 결과 반환
    }

    // 외부 버퍼 할당
    Ipp8u* pBuffer = ippsMalloc_8u(bufferSize);
    if (pBuffer == nullptr) {
        std::cerr << "Error: Failed to allocate buffer." << std::endl;
        return result; // 빈 결과 반환
    }

    // 가우시안 필터 스펙트럼 구조체 메모리 할당
    IppFilterGaussianSpec* pSpec = reinterpret_cast<IppFilterGaussianSpec*>(ippsMalloc_8u(specSize));
    if (pSpec == nullptr) {
        std::cerr << "Error: Failed to allocate spec structure." << std::endl;
        ippsFree(pBuffer);
        return result; // 빈 결과 반환
    }

    // 가우시안 필터 초기화 (표준 편차는 예시로 1.5로 설정)
    float sigma = 1.5f;
    status = ippiFilterGaussianInit(roiSize, kernelSize, sigma, ippBorderRepl, ipp16u, 3, pSpec, pBuffer);
    if (status != ippStsNoErr) {
        std::cerr << "Error: ippiFilterGaussianInit failed with status " << status << std::endl;
        ippsFree(pBuffer);
        ippsFree(pSpec);
        return result; // 빈 결과 반환
    }

    // 가우시안 필터 적용
    int srcStep = bgrImage.cols * sizeof(Ipp16u) * 3;
    int dstStep = outputImage.cols * sizeof(Ipp16u) * 3;
    Ipp16u borderValue[3] = { 0, 0, 0 }; // 가우시안 필터 적용 시 사용할 보더 값
    status = ippiFilterGaussianBorder_16u_C3R(pSrc, srcStep, pDst, dstStep, roiSize, borderValue, pSpec, pBuffer);
    if (status != ippStsNoErr) {
        std::cerr << "Error: ippiFilterGaussianBorder_16u_C3R failed with status " << status << std::endl;
        ippsFree(pBuffer);
        ippsFree(pSpec);
        return result; // 빈 결과 반환
    }

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    // 8비트 이미지로 변환
    cv::Mat finalOutputImage;
    outputImage.convertTo(finalOutputImage, CV_8UC3, 1.0 / 256.0);

    // 결과 설정
    result = setResult(result, inputImage, finalOutputImage, "gaussianBlur", "IPP", elapsedTimeMs);

    // 메모리 해제
    ippsFree(pBuffer);
    ippsFree(pSpec);

    return result;
}


ImageProcessor::ProcessingResult ImageProcessor::gaussianBlurCUDA(cv::Mat& inputImage, int kernelSize)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

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

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "gaussianBlur", "CUDA", elapsedTimeMs);

    return result;

}

ImageProcessor::ProcessingResult ImageProcessor::gaussianBlurCUDAKernel(cv::Mat& inputImage, int kernelSize)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    cv::Mat outputImage;
    callGaussianBlurCUDA(inputImage, outputImage, kernelSize);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "gaussianBlur", "CUDAKernel", elapsedTimeMs);

    return result;
}



//Canny
QFuture<bool> ImageProcessor::cannyEdges(cv::Mat& imageOpenCV
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

            ProcessingResult outputOpenCV = cannyEdgesOpenCV(imageOpenCV);
            lastProcessedImageOpenCV = outputOpenCV.processedImage.clone();
            results.append(outputOpenCV);

            ProcessingResult outputIPP = cannyEdgesIPP(imageIPP);
            lastProcessedImageIPP = outputIPP.processedImage.clone();
            results.append(outputIPP);

            ProcessingResult outputCUDA = cannyEdgesCUDA(imageCUDA);
            lastProcessedImageCUDA = outputCUDA.processedImage.clone();
            results.append(outputCUDA);

            ProcessingResult outputCUDAKernel = cannyEdgesCUDAKernel(imageCUDAKernel);
            lastProcessedImageCUDAKernel = outputCUDAKernel.processedImage.clone();
            results.append(outputCUDAKernel);            

            emit imageProcessed(results);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "Exception occurred while applying Canny edges:" << e.what();
            return false;
        }
        });
}

ImageProcessor::ProcessingResult ImageProcessor::cannyEdgesOpenCV(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    cv::Mat grayImage;
    if (inputImage.channels() == 3) {
        grayImage = convertToGrayOpenCV(inputImage);
    }
    else {
        grayImage = inputImage.clone();
    }

    cv::Mat outputImage;
    cv::Canny(grayImage, outputImage, 50, 150);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "cannyEdges", "OpenCV", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::cannyEdgesIPP(cv::Mat& inputImage) {
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    cv::Mat grayImage;
    if (inputImage.channels() == 3) {
        grayImage = convertToGrayIPP(inputImage);
    } else {
        grayImage = inputImage.clone();
    }    

    // IPP를 사용하여 Canny 엣지 감지 수행
    IppiSize roiSize = { grayImage.cols, grayImage.rows };
    int srcStep = grayImage.step;
    int dstStep = grayImage.cols;
    Ipp8u* srcData = grayImage.data;
    Ipp8u* dstData = ippsMalloc_8u(roiSize.width * roiSize.height); // 출력 이미지 메모리 할당

    if (!dstData) {
        std::cerr << "Memory allocation error: Failed to allocate dstData" << std::endl;
        return result; // 메모리 할당 오류 처리 중단
    }

    // IPP Canny 처리를 위한 임시 버퍼 크기 계산
    int bufferSize;
    IppStatus status = ippiCannyBorderGetSize(roiSize, ippFilterSobel, ippMskSize3x3, ipp8u, &bufferSize);
    if (status != ippStsNoErr) {
        std::cerr << "IPP error: Failed to calculate buffer size for Canny edge detection (" << status << ")";
        ippsFree(dstData); // 할당된 메모리 해제
        return result; // 오류 발생 시 처리 중단
    }

    // 임시 버퍼 할당
    Ipp8u* pBuffer = ippsMalloc_8u(bufferSize);
    if (!pBuffer) {
        std::cerr << "Memory allocation error: Failed to allocate dstData" << std::endl;
        ippsFree(dstData); // 이미 할당된 dstData 메모리도 해제
        return result; // 메모리 할당 오류 처리 중단
    }

    // IPP Canny 엣지 감지 수행
    status = ippiCannyBorder_8u_C1R(srcData, srcStep, dstData, dstStep, roiSize, ippFilterSobel, ippMskSize3x3, ippBorderRepl, 0, 50.0f, 150.0f, ippNormL2, pBuffer);
    if (status != ippStsNoErr) {
        std::cerr << "IPP error: Failed to perform Canny edge detection (" << status << ")";
        ippsFree(pBuffer); // 할당된 메모리 해제
        ippsFree(dstData); // 할당된 메모리 해제
        return result; // 오류 발생 시 처리 중단
    }


    // 결과를 OpenCV Mat 형식으로 변환
    cv::Mat outputImage(grayImage.rows, grayImage.cols, CV_8UC1, dstData);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 경과 시간 계산

    result = setResult(result, inputImage, outputImage, "cannyEdges", "IPP", elapsedTimeMs);

    // 할당된 메모리 해제
    ippsFree(pBuffer);
    ippsFree(dstData);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::cannyEdgesCUDA(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    cv::Mat grayImage;
    if (inputImage.channels() == 3) {
        grayImage = convertToGrayCUDA(inputImage);
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

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "cannyEdges", "CUDA", elapsedTimeMs);

    // GPU 메모리 해제는 GpuMat 객체가 스코프를 벗어날 때 자동으로 처리됩니다.

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::cannyEdgesCUDAKernel(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    cv::Mat grayImage;
    if (inputImage.channels() == 3) {
        grayImage = convertToGrayCUDAKernel(inputImage);
    }
    else {
        grayImage = inputImage.clone();
    }

    cv::Mat outputImage;
    callCannyEdgesCUDA(grayImage, outputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, grayImage, outputImage, "cannyEdges", "CUDAKernel", elapsedTimeMs);

    return result;
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

        QString outputInfoOpenCV = "(Output) Channels: " + QString::number(lastProcessedImageOpenCV.channels())
            + ", type: " + QString::number(lastProcessedImageOpenCV.type())
            + ", depth: " + QString::number(lastProcessedImageOpenCV.depth());
        QString outputInfoIPP = "(Output) Channels: " + QString::number(lastProcessedImageIPP.channels())
            + ", type: " + QString::number(lastProcessedImageIPP.type())
            + ", depth: " + QString::number(lastProcessedImageIPP.depth());
        QString outputInfoCUDA = "(Output) Channels: " + QString::number(lastProcessedImageCUDA.channels())
            + ", type: " + QString::number(lastProcessedImageCUDA.type())
            + ", depth: " + QString::number(lastProcessedImageCUDA.depth());
        QString outputInfoCUDAKernel = "(Output) Channels: " + QString::number(lastProcessedImageCUDAKernel.channels())
            + ", type: " + QString::number(lastProcessedImageCUDAKernel.type())
            + ", depth: " + QString::number(lastProcessedImageCUDAKernel.depth());

        double endTime = cv::getTickCount(); // 종료 시간 측정
        double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

        // 결과 생성
        results.append(ProcessingResult(functionName, "OpenCV", lastProcessedImageOpenCV.clone(), elapsedTimeMs, "", outputInfoOpenCV));
        results.append(ProcessingResult(functionName, "IPP", lastProcessedImageIPP.clone(), elapsedTimeMs, "", outputInfoIPP));
        results.append(ProcessingResult(functionName, "CUDA", lastProcessedImageCUDA.clone(), elapsedTimeMs, "", outputInfoCUDA));
        results.append(ProcessingResult(functionName, "CUDAKernel", lastProcessedImageCUDAKernel.clone(), elapsedTimeMs, "", outputInfoCUDAKernel));

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

        QString outputInfoOpenCV = "(Output) Channels: " + QString::number(lastProcessedImageOpenCV.channels())
            + ", type: " + QString::number(lastProcessedImageOpenCV.type())
            + ", depth: " + QString::number(lastProcessedImageOpenCV.depth());
        QString outputInfoIPP = "(Output) Channels: " + QString::number(lastProcessedImageIPP.channels())
            + ", type: " + QString::number(lastProcessedImageIPP.type())
            + ", depth: " + QString::number(lastProcessedImageIPP.depth());
        QString outputInfoCUDA = "(Output) Channels: " + QString::number(lastProcessedImageCUDA.channels())
            + ", type: " + QString::number(lastProcessedImageCUDA.type())
            + ", depth: " + QString::number(lastProcessedImageCUDA.depth());
        QString outputInfoCUDAKernel = "(Output) Channels: " + QString::number(lastProcessedImageCUDAKernel.channels())
            + ", type: " + QString::number(lastProcessedImageCUDAKernel.type())
            + ", depth: " + QString::number(lastProcessedImageCUDAKernel.depth());

        double endTime = cv::getTickCount(); // 종료 시간 측정
        double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

        // 결과 생성
        results.append(ProcessingResult(functionName, "OpenCV", lastProcessedImageOpenCV.clone(), elapsedTimeMs, "", outputInfoOpenCV));
        results.append(ProcessingResult(functionName, "IPP", lastProcessedImageIPP.clone(), elapsedTimeMs, "", outputInfoIPP));
        results.append(ProcessingResult(functionName, "CUDA", lastProcessedImageCUDA.clone(), elapsedTimeMs, "", outputInfoCUDA));
        results.append(ProcessingResult(functionName, "CUDAKernel", lastProcessedImageCUDAKernel.clone(), elapsedTimeMs, "", outputInfoCUDAKernel));

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

ImageProcessor::ProcessingResult ImageProcessor::setResult(ProcessingResult& result, cv::Mat& inputImage, cv::Mat& outputImage, QString functionName, QString processName, double processingTime)
{
    result.functionName = functionName;
    result.processName = processName;
    result.inputInfo = "\n(Input) Channels: " + QString::number(inputImage.channels())
        + ", type: " + QString::number(inputImage.type())
        + "(" + ImageTypeConverter::getImageTypeString(inputImage.type()) + ")"
        + ", depth: " + QString::number(inputImage.depth());
    result.processedImage = outputImage.clone();
    result.processingTime = processingTime;
    result.outputInfo = "\n(Output) Channels: " + QString::number(outputImage.channels())
        + ", type: " + QString::number(outputImage.type())
        + "(" + ImageTypeConverter::getImageTypeString(outputImage.type()) + ")"
        + ", depth: " + QString::number(outputImage.depth());

    return result;
}
