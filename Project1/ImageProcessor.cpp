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
                                        , cv::Mat& imageCUDAKernel
                                        , cv::Mat& imageNPP
                                        , cv::Mat& imageGStreamer)
{
    //함수 이름을 문자열로 저장
    const char* functionName = __func__;

    return QtConcurrent::run([this
        , &imageOpenCV
        , &imageIPP
        , &imageCUDA
        , &imageCUDAKernel
        , &imageNPP
        , &imageGStreamer
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
            pushToUndoStackNPP(imageNPP.clone());
            pushToUndoStackGStreamer(imageGStreamer.clone());

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

            ProcessingResult outputNPP = rotateNPP(imageNPP);
            lastProcessedImageNPP = outputNPP.processedImage.clone();
            results.append(outputNPP);

            ProcessingResult outputGStreamer = rotateGStreamer(imageGStreamer);
            lastProcessedImageGStreamer = outputGStreamer.processedImage.clone();
            results.append(outputGStreamer);

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
                                            , cv::Mat& imageNPP
                                            , cv::Mat& imageGStreamer
                                            , double scaleFactor)
{
    //함수 이름을 문자열로 저장
    const char* functionName = __func__;

    return QtConcurrent::run([this
                            , &imageOpenCV
                            , &imageIPP
                            , &imageCUDA
                            , &imageCUDAKernel
                            , &imageNPP
                            , &imageGStreamer
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
                pushToUndoStackNPP(imageNPP.clone());
                pushToUndoStackGStreamer(imageGStreamer.clone());

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

                ProcessingResult outputNPP = zoomNPP(imageNPP, newWidth, newHeight);
                lastProcessedImageNPP = outputNPP.processedImage.clone();
                results.append(outputNPP);

                ProcessingResult outputGStreamer = zoomGStreamer(imageGStreamer, newWidth, newHeight);
                lastProcessedImageGStreamer = outputGStreamer.processedImage.clone();
                results.append(outputGStreamer);

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
                                        , cv::Mat& imageNPP
                                        , cv::Mat& imageGStreamer
                                        , double scaleFactor)
{
    //함수 이름을 문자열로 저장
    const char* functionName = __func__;

    return QtConcurrent::run([this
                            , &imageOpenCV
                            , &imageIPP
                            , &imageCUDA
                            , &imageCUDAKernel
                            , &imageNPP
                            , &imageGStreamer
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
            pushToUndoStackNPP(imageNPP.clone());
            pushToUndoStackGStreamer(imageGStreamer.clone());

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

            ProcessingResult outputNPP = zoomNPP(imageNPP, newWidth, newHeight);
            lastProcessedImageNPP = outputNPP.processedImage.clone();
            results.append(outputNPP);

            ProcessingResult outputGStreamer = zoomGStreamer(imageGStreamer, newWidth, newHeight);
            lastProcessedImageGStreamer = outputGStreamer.processedImage.clone();
            results.append(outputGStreamer);

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
                                        , cv::Mat& imageCUDAKernel
                                        , cv::Mat& imageNPP
                                        , cv::Mat& imageGStreamer)
{
    const char* functionName = __func__;

    return QtConcurrent::run([this
                            , &imageOpenCV
                            , &imageIPP
                            , &imageCUDA
                            , &imageCUDAKernel
                            , &imageNPP
                            , &imageGStreamer
                            , functionName]() -> bool {
        
        QMutexLocker locker(&mutex);

        try {
            if (imageOpenCV.channels() != 3 && imageOpenCV.channels() != 1) {
                qDebug() << "Input image must be a 3-channel BGR image or already grayscale.";
                return false;
            }

            pushToUndoStackOpenCV(imageOpenCV.clone());
            pushToUndoStackIPP(imageIPP.clone());
            pushToUndoStackCUDA(imageCUDA.clone());
            pushToUndoStackCUDAKernel(imageCUDAKernel.clone());
            pushToUndoStackNPP(imageNPP.clone());
            pushToUndoStackGStreamer(imageGStreamer.clone());

            if (imageOpenCV.channels() == 3) {                

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

                ProcessingResult outputNPP = grayScaleNPP(imageNPP);
                lastProcessedImageNPP = outputNPP.processedImage.clone();
                results.append(outputNPP);

                ProcessingResult outputGStreamer = grayScaleGStreamer(imageGStreamer);
                lastProcessedImageGStreamer = outputGStreamer.processedImage.clone();
                results.append(outputGStreamer);

                emit imageProcessed(results);
            }
            else {

                lastProcessedImageOpenCV = imageOpenCV.clone();
                lastProcessedImageIPP = imageIPP.clone();
                lastProcessedImageCUDA = imageCUDA.clone();
                lastProcessedImageCUDAKernel = imageCUDAKernel.clone();
                lastProcessedImageNPP = imageNPP.clone();
                lastProcessedImageGStreamer = imageGStreamer.clone();
            }

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "Exception occurred while converting to grayscale:" << e.what();
            return false;
        }
        });
}

ImageProcessor::ProcessingResult ImageProcessor::grayScaleOpenCV(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorOpenCV IPOpenCV;
    cv::Mat outputImage = IPOpenCV.grayScale(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "grayScale", "OpenCV", elapsedTimeMs, "");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::grayScaleIPP(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorIPP IPIPP;
    cv::Mat outputImage = IPIPP.grayScale(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산   

    result = setResult(result, inputImage, outputImage, "grayScale", "IPP", elapsedTimeMs, "");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::grayScaleCUDA(cv::Mat& inputImage)
{    
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorCUDA IPCUDA;
    cv::Mat outputImage = IPCUDA.grayScale(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "grayScale", "CUDA", elapsedTimeMs, "");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::grayScaleCUDAKernel(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorCUDAKernel IPCUDAK;
    cv::Mat outputImage = IPCUDAK.grayScale(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "grayScale", "CUDAKernel", elapsedTimeMs, "");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::grayScaleNPP(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorNPP IPNPP;
    cv::Mat outputImage = IPNPP.grayScale(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "grayScale", "NPP", elapsedTimeMs
        , "");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::grayScaleGStreamer(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorGStreamer IPGStreamer;
    cv::Mat outputImage = IPGStreamer.grayScale(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "grayScale", "GStreamer", elapsedTimeMs
        , "");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::zoomOpenCV(cv::Mat& inputImage, double newWidth, double newHeight)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorOpenCV IPOpenCV;
    cv::Mat outputImage;
    outputImage = IPOpenCV.zoom(inputImage, newWidth, newHeight, 0, 0, cv::INTER_LINEAR);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "zoom", "OpenCV", elapsedTimeMs
        , QString("w:%1, h:%2, 0, 0, cv::INTER_LINEAR").arg(newWidth).arg(newHeight));

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::zoomIPP(cv::Mat& inputImage, double newWidth, double newHeight) {
 
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorIPP IPIPP;
    cv::Mat outputImage = IPIPP.zoom(inputImage, newWidth, newHeight);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "zoom", "IPP", elapsedTimeMs
    ,QString("w:%1, h:%2").arg(newWidth).arg(newHeight));

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::zoomCUDA(cv::Mat& inputImage, double newWidth, double newHeight)
{    
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorCUDA IPCUDA;
    cv::Mat outputImage = IPCUDA.zoom(inputImage, newWidth, newHeight);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "zoom", "CUDA", elapsedTimeMs
    , QString("w:%1, h:%2").arg(newWidth).arg(newHeight));

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::zoomCUDAKernel(cv::Mat& inputImage, double newWidth, double newHeight)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorCUDAKernel IPCUDAK;
    cv::Mat outputImage = IPCUDAK.zoom(inputImage, newWidth, newHeight);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "zoom", "CUDAKernel", elapsedTimeMs
    , QString("w:%1, h:%2").arg(newWidth).arg(newHeight));

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::zoomNPP(cv::Mat& inputImage, double newWidth, double newHeight) {
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorNPP IPNPP;
    cv::Mat outputImage = IPNPP.zoom(inputImage, newWidth, newHeight);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "zoom", "NPP", elapsedTimeMs
        , QString("w:%1, h:%2").arg(newWidth).arg(newHeight));

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::zoomGStreamer(cv::Mat& inputImage, double newWidth, double newHeight) {
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorNPP IPNPP;
    cv::Mat outputImage = IPNPP.zoom(inputImage, newWidth, newHeight);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "zoom", "GStreamer", elapsedTimeMs
        , QString("w:%1, h:%2").arg(newWidth).arg(newHeight));

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::rotateOpenCV(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorOpenCV IPOpenCV;
    cv::Mat outputImage = IPOpenCV.rotate(inputImage, true);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "rotate", "OpenCV", elapsedTimeMs
        , "angle: R 90");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::rotateIPP(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount();

    ImageProcessorIPP IPIPP;
    cv::Mat outputImage = IPIPP.rotate(inputImage, true);//90.0 오른쪽, 270.0 왼쪽

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "rotate", "IPP", elapsedTimeMs
    , "angle: R 90");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::rotateCUDA(cv::Mat& inputImage)
{
    ProcessingResult result;

    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorCUDA IPCUDA;
    cv::Mat outputImage = IPCUDA.rotate(inputImage, true); // 270.0 : 오른쪽 90도, 90.0 : 왼쪽 90도   

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "rotate", "CUDA", elapsedTimeMs
    ,"angle: R 90");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::rotateCUDAKernel(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorCUDAKernel IPCUDAK;
    cv::Mat outputImage = IPCUDAK.rotate(inputImage, true); //true:right, false:left

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "rotate", "CUDAKernel", elapsedTimeMs
    , "angle: R 90");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::rotateNPP(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorNPP IPNPP;
    cv::Mat outputImage = IPNPP.rotate(inputImage, true); //true:right, false:left

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "rotate", "NPP", elapsedTimeMs
        , "angle: R 90");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::rotateGStreamer(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorGStreamer IPGStreamer;
    cv::Mat outputImage = IPGStreamer.rotate(inputImage, true); //true:right, false:left

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "rotate", "GStreamer", elapsedTimeMs
        , "angle: R 90");

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
                                            , cv::Mat& imageNPP
                                            , cv::Mat& imageGStreamer
                                            , int kernelSize)
{
    //함수 이름을 문자열로 저장
    const char* functionName = __func__;

    return QtConcurrent::run([this
        , &imageOpenCV
        , &imageIPP
        , &imageCUDA
        , &imageCUDAKernel
        , &imageNPP
        , &imageGStreamer
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
            pushToUndoStackNPP(imageNPP.clone());
            pushToUndoStackGStreamer(imageGStreamer.clone());

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

            ProcessingResult outputNPP = gaussianBlurNPP(imageNPP, kernelSize);
            lastProcessedImageNPP = outputNPP.processedImage.clone();
            results.append(outputNPP);

            ProcessingResult outputGStreamer = gaussianBlurGStreamer(imageGStreamer, kernelSize);
            lastProcessedImageGStreamer = outputGStreamer.processedImage.clone();
            results.append(outputGStreamer);

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

    ImageProcessorOpenCV IPOpenCV;
    cv::Mat outputImage = IPOpenCV.gaussianBlur(inputImage, kernelSize, 0, 0, 1);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "gaussianBlur", "OpenCV", elapsedTimeMs
    , QString("kSize:%1, 0, 0, 1 ").arg(kernelSize));

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::gaussianBlurIPP(cv::Mat& inputImage, int kernelSize) {
    
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorIPP IPIPP;
    cv::Mat outputImage = IPIPP.gaussianBlur(inputImage, kernelSize);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "gaussianBlur", "IPP", elapsedTimeMs
        , QString("kSize:%1").arg(kernelSize));
    
    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::gaussianBlurCUDA(cv::Mat& inputImage, int kernelSize)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorCUDA IPCUDA;
    cv::Mat outputImage = IPCUDA.gaussianBlur(inputImage, kernelSize);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "gaussianBlur", "CUDA", elapsedTimeMs
        , QString("kSize:%1").arg(kernelSize));

    return result;

}

ImageProcessor::ProcessingResult ImageProcessor::gaussianBlurCUDAKernel(cv::Mat& inputImage, int kernelSize)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorCUDAKernel IPCUDAK;
    cv::Mat outputImage = IPCUDAK.gaussianBlur(inputImage, kernelSize);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "gaussianBlur", "CUDAKernel", elapsedTimeMs
        , QString("kSize:%1").arg(kernelSize));

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::gaussianBlurNPP(cv::Mat& inputImage, int kernelSize)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorNPP IPNPP;
    cv::Mat outputImage = IPNPP.gaussianBlur(inputImage, kernelSize);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "gaussianBlur", "NPP", elapsedTimeMs
        , QString("kSize:%1").arg(kernelSize));

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::gaussianBlurGStreamer(cv::Mat& inputImage, int kernelSize)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorGStreamer IPGStreamer;
    cv::Mat outputImage = IPGStreamer.gaussianBlur(inputImage, kernelSize);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "gaussianBlur", "GStreamer", elapsedTimeMs
        , QString("kSize:%1").arg(kernelSize));

    return result;
}

//Canny
QFuture<bool> ImageProcessor::cannyEdges(cv::Mat& imageOpenCV
                                        , cv::Mat& imageIPP
                                        , cv::Mat& imageCUDA
                                        , cv::Mat& imageCUDAKernel
                                        , cv::Mat& imageNPP
                                        , cv::Mat& imageGStreamer)
{
    //함수 이름을 문자열로 저장
    const char* functionName = __func__;

    return QtConcurrent::run([this
        , &imageOpenCV
        , &imageIPP
        , &imageCUDA
        , &imageCUDAKernel
        , &imageNPP
        , &imageGStreamer
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
            pushToUndoStackNPP(imageNPP.clone());
            pushToUndoStackGStreamer(imageGStreamer.clone());

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

            ProcessingResult outputNPP = cannyEdgesNPP(imageNPP);
            lastProcessedImageNPP = outputNPP.processedImage.clone();
            results.append(outputNPP);

            ProcessingResult outputGStreamer = cannyEdgesGStreamer(imageGStreamer);
            lastProcessedImageGStreamer = outputGStreamer.processedImage.clone();
            results.append(outputGStreamer);

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

    ImageProcessorOpenCV IPOpenCV;
    cv::Mat outputImage = IPOpenCV.cannyEdges(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "cannyEdges", "OpenCV", elapsedTimeMs
        , "thresold1:50, thresold2:150");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::cannyEdgesIPP(cv::Mat& inputImage) {
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorIPP IPIPP;
    cv::Mat outputImage = IPIPP.cannyEdges(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 경과 시간 계산

    result = setResult(result, inputImage, outputImage, "cannyEdges", "IPP", elapsedTimeMs
        , "");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::cannyEdgesCUDA(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorCUDA IPCUDA;
    cv::Mat outputImage = IPCUDA.cannyEdges(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "cannyEdges", "CUDA", elapsedTimeMs
        , "");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::cannyEdgesCUDAKernel(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorCUDAKernel IPCUDAK;
    cv::Mat outputImage = IPCUDAK.cannyEdges(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "cannyEdges", "CUDAKernel", elapsedTimeMs
    , "");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::cannyEdgesNPP(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorNPP IPNPP;
    cv::Mat outputImage = IPNPP.cannyEdges(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "cannyEdges", "NPP", elapsedTimeMs
        , "");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::cannyEdgesGStreamer(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorNPP IPGStreamer;
    cv::Mat outputImage = IPGStreamer.cannyEdges(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "cannyEdges", "GStreamer", elapsedTimeMs
        , "");

    return result;
}

QFuture<bool> ImageProcessor::medianFilter(cv::Mat& imageOpenCV
                                            , cv::Mat& imageIPP
                                            , cv::Mat& imageCUDA
                                            , cv::Mat& imageCUDAKernel
                                            , cv::Mat& imageNPP
                                            , cv::Mat& imageGStreamer)
{

    //함수 이름을 문자열로 저장
    const char* functionName = __func__;

    return QtConcurrent::run([this
        , &imageOpenCV
        , &imageIPP
        , &imageCUDA
        , &imageCUDAKernel
        , &imageNPP
        , &imageGStreamer
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
            pushToUndoStackNPP(imageNPP.clone());
            pushToUndoStackGStreamer(imageGStreamer.clone());

            QVector<ProcessingResult> results;

            ProcessingResult outputOpenCV = medianFilterOpenCV(imageOpenCV);
            lastProcessedImageOpenCV = outputOpenCV.processedImage.clone();
            results.append(outputOpenCV);

            ProcessingResult outputIPP = medianFilterIPP(imageIPP);
            lastProcessedImageIPP = outputIPP.processedImage.clone();
            results.append(outputIPP);

            ProcessingResult outputCUDA = medianFilterCUDA(imageCUDA);
            lastProcessedImageCUDA = outputCUDA.processedImage.clone();
            results.append(outputCUDA);

            ProcessingResult outputCUDAKernel = medianFilterCUDAKernel(imageCUDAKernel);
            lastProcessedImageCUDAKernel = outputCUDAKernel.processedImage.clone();
            results.append(outputCUDAKernel);

            ProcessingResult outputNPP = medianFilterNPP(imageNPP);
            lastProcessedImageNPP = outputNPP.processedImage.clone();
            results.append(outputNPP);

            ProcessingResult outputGStreamer = medianFilterGStreamer(imageGStreamer);
            lastProcessedImageGStreamer = outputGStreamer.processedImage.clone();
            results.append(outputGStreamer);

            emit imageProcessed(results);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "median 필터 적용 중 오류 발생: "
                << e.what();
            return false;
        }
        });
}

ImageProcessor::ProcessingResult ImageProcessor::medianFilterOpenCV(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorOpenCV IPOpenCV;
    cv::Mat outputImage = IPOpenCV.medianFilter(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "medianFilter", "OpenCV", elapsedTimeMs
    , "ksize:5");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::medianFilterIPP(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorIPP IPIPP;
    cv::Mat outputImage = IPIPP.medianFilter(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "medianFilter", "IPP", elapsedTimeMs
    , "ksize:5");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::medianFilterCUDA(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorCUDA IPCUDA;
    cv::Mat outputImage = IPCUDA.medianFilter(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "medianFilter", "CUDA", elapsedTimeMs
    , "ksize:5");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::medianFilterCUDAKernel(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorCUDAKernel IPCUDAK;
    cv::Mat outputImage = IPCUDAK.medianFilter(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "medianFilter", "OpenCV", elapsedTimeMs
    , "ksize:5");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::medianFilterNPP(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorNPP IPNPP;
    cv::Mat outputImage = IPNPP.medianFilter(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "medianFilter", "NPP", elapsedTimeMs
        , "ksize:5");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::medianFilterGStreamer(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorGStreamer IPGStreamer;
    cv::Mat outputImage = IPGStreamer.medianFilter(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "medianFilter", "GStreamer", elapsedTimeMs
        , "ksize:5");

    return result;
}

QFuture<bool> ImageProcessor::laplacianFilter(cv::Mat& imageOpenCV
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

            ProcessingResult outputOpenCV = laplacianFilterOpenCV(imageOpenCV);
            lastProcessedImageOpenCV = outputOpenCV.processedImage.clone();
            results.append(outputOpenCV);

            ProcessingResult outputIPP = laplacianFilterIPP(imageOpenCV);
            lastProcessedImageIPP = outputIPP.processedImage.clone();
            results.append(outputIPP);

            ProcessingResult outputCUDA = laplacianFilterCUDA(imageCUDA);
            lastProcessedImageCUDA = outputCUDA.processedImage.clone();
            results.append(outputCUDA);

            ProcessingResult outputCUDAKernel = laplacianFilterCUDAKernel(imageCUDAKernel);
            lastProcessedImageCUDAKernel = outputCUDAKernel.processedImage.clone();
            results.append(outputCUDAKernel);                  

            emit imageProcessed(results);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "laplacian 필터 적용 중 오류 발생: "
                << e.what();
            return false;
        }
        });
}

ImageProcessor::ProcessingResult ImageProcessor::laplacianFilterOpenCV(cv::Mat& inputImage)
{
    std::cout << "laplacianFilter OpenCV" << std::endl;

    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorOpenCV IPOpenCV;
    cv::Mat outputImage = IPOpenCV.laplacianFilter(inputImage);    

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "laplacianFilter", "OpenCV", elapsedTimeMs
    , "");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::laplacianFilterIPP(cv::Mat& inputImage)
{
    std::cout << "laplacianFilter IPP" << std::endl;

    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorIPP IPIPP;
    cv::Mat outputImage = IPIPP.laplacianFilter(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "laplacianFilter", "IPP", elapsedTimeMs
    , "");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::laplacianFilterCUDA(cv::Mat& inputImage)
{
    std::cout << "laplacianFilter CUDA" << std::endl;

    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorCUDA IPCUDA;
    cv::Mat outputImage = IPCUDA.laplacianFilter(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "laplacianFilter", "CUDA", elapsedTimeMs
    , "");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::laplacianFilterCUDAKernel(cv::Mat& inputImage)
{
    std::cout << "laplacianFilter CUDAKernel" << std::endl;

    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorCUDAKernel IPCUDAK;
    cv::Mat outputImage = IPCUDAK.laplacianFilter(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "laplacianFilter", "CUDAKernel", elapsedTimeMs
    , "");

    return result;
}


QFuture<bool> ImageProcessor::bilateralFilter(cv::Mat& imageOpenCV
                                            , cv::Mat& imageIPP
                                            , cv::Mat& imageCUDA
                                            , cv::Mat& imageCUDAKernel
                                            , cv::Mat& imageNPP
                                            , cv::Mat& imageGStreamer)
{
    //함수 이름을 문자열로 저장
    const char* functionName = __func__;

    return QtConcurrent::run([this
        , &imageOpenCV
        , &imageIPP
        , &imageCUDA
        , &imageCUDAKernel
        , &imageNPP
        , &imageGStreamer
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
            pushToUndoStackNPP(imageNPP.clone());
            pushToUndoStackGStreamer(imageGStreamer.clone());

            QVector<ProcessingResult> results;

            ProcessingResult outputOpenCV = bilateralFilterOpenCV(imageOpenCV);
            lastProcessedImageOpenCV = outputOpenCV.processedImage.clone();
            results.append(outputOpenCV);

            ProcessingResult outputIPP = bilateralFilterIPP(imageIPP);
            lastProcessedImageIPP = outputIPP.processedImage.clone();
            results.append(outputIPP);

            ProcessingResult outputCUDA = bilateralFilterCUDA(imageCUDA);
            lastProcessedImageCUDA = outputCUDA.processedImage.clone();
            results.append(outputCUDA);

            ProcessingResult outputCUDAKernel = bilateralFilterCUDAKernel(imageIPP);
            lastProcessedImageCUDAKernel = outputCUDAKernel.processedImage.clone();
            results.append(outputCUDAKernel);   

            ProcessingResult outputNPP = bilateralFilterNPP(imageNPP);
            lastProcessedImageNPP = outputNPP.processedImage.clone();
            results.append(outputNPP);

            ProcessingResult outputGStreamer = bilateralFilterGStreamer(imageGStreamer);
            lastProcessedImageGStreamer = outputGStreamer.processedImage.clone();
            results.append(outputGStreamer);

            emit imageProcessed(results);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "bilateral 필터 적용 중 오류 발생: "
                << e.what();
            return false;
        }
        });
}

ImageProcessor::ProcessingResult ImageProcessor::bilateralFilterOpenCV(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorOpenCV IPOpenCV;
    cv::Mat outputImage = IPOpenCV.bilateralFilter(inputImage);   

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "bilateralFilter", "OpenCV", elapsedTimeMs
    , "9, 75, 75, cv::BORDER_DEFAULT");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::bilateralFilterIPP(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorIPP IPIPP;
    cv::Mat outputImage = IPIPP.bilateralFilter(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "bilateralFilter", "IPP", elapsedTimeMs
    , "9, 75, 75");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::bilateralFilterCUDA(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorCUDA IPCUDA;
    cv::Mat outputImage = IPCUDA.bilateralFilter(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "bilateralFilter", "CUDA", elapsedTimeMs
    , "9, 75, 75");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::bilateralFilterCUDAKernel(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorCUDAKernel IPCUDAK;
    cv::Mat outputImage = IPCUDAK.bilateralFilter(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "bilateralFilter", "CUDAKernel", elapsedTimeMs
    , "9, 75, 75");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::bilateralFilterNPP(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorNPP IPNPP;
    cv::Mat outputImage = IPNPP.bilateralFilter(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "bilateralFilter", "NPP", elapsedTimeMs
        , "9, 75, 75, NPP_BORDER_REPLICATE");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::bilateralFilterGStreamer(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorGStreamer IPGStreamer;
    cv::Mat outputImage = IPGStreamer.bilateralFilter(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "bilateralFilter", "GStreamer", elapsedTimeMs
        , "9, 75, 75, cv::BORDER_DEFAULT");

    return result;
}

QFuture<bool> ImageProcessor::sobelFilter(cv::Mat& imageOpenCV
                                        , cv::Mat& imageIPP
                                        , cv::Mat& imageCUDA
                                        , cv::Mat& imageCUDAKernel
                                        , cv::Mat& imageNPP
                                        , cv::Mat& imageGStreamer)
{
    //함수 이름을 문자열로 저장
    const char* functionName = __func__;

    return QtConcurrent::run([this
        , &imageOpenCV
        , &imageIPP
        , &imageCUDA
        , &imageCUDAKernel
        , &imageNPP
        , &imageGStreamer
        , functionName]() -> bool {

        if (imageOpenCV.empty()) {
            qDebug() << "Input image is empty.";
            return false;
        }

        pushToUndoStackOpenCV(imageOpenCV.clone());
        pushToUndoStackIPP(imageIPP.clone());
        pushToUndoStackCUDA(imageCUDA.clone());
        pushToUndoStackCUDAKernel(imageCUDAKernel.clone());
        pushToUndoStackNPP(imageNPP.clone());
        pushToUndoStackGStreamer(imageGStreamer.clone());

        QVector<ProcessingResult> results;

        ProcessingResult outputOpenCV = sobelFilterOpenCV(imageOpenCV);
        lastProcessedImageOpenCV = outputOpenCV.processedImage.clone();
        results.append(outputOpenCV);

        ProcessingResult outputIPP = sobelFilterIPP(imageIPP);
        lastProcessedImageIPP = outputIPP.processedImage.clone();
        results.append(outputIPP);

        ProcessingResult outputCUDA = sobelFilterCUDA(imageCUDA);
        lastProcessedImageCUDA = outputCUDA.processedImage.clone();
        results.append(outputCUDA);

        ProcessingResult outputCUDAKernel = sobelFilterCUDAKernel(imageCUDAKernel);
        lastProcessedImageCUDAKernel = outputCUDAKernel.processedImage.clone();
        results.append(outputCUDAKernel);

        ProcessingResult outputNPP = sobelFilterNPP(imageNPP);
        lastProcessedImageNPP = outputNPP.processedImage.clone();
        results.append(outputNPP);

        ProcessingResult outputGStreamer = sobelFilterGStreamer(imageGStreamer);
        lastProcessedImageGStreamer = outputGStreamer.processedImage.clone();
        results.append(outputGStreamer);

        emit imageProcessed(results);

        return true;
        });
}

ImageProcessor::ProcessingResult ImageProcessor::sobelFilterOpenCV(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorOpenCV IPOpenCV;
    cv::Mat outputImage = IPOpenCV.sobelFilter(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "sobelFilter", "OpenCV", elapsedTimeMs
    , "");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::sobelFilterIPP(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorIPP IPIPP;
    cv::Mat outputImage = IPIPP.sobelFilter(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "sobelFilter", "IPP", elapsedTimeMs
    , "");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::sobelFilterCUDA(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorCUDA IPCUDA;
    cv::Mat outputImage = IPCUDA.sobelFilter(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "sobelFilter", "CUDA", elapsedTimeMs
    , "");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::sobelFilterCUDAKernel(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorCUDAKernel IPCUDAK;
    cv::Mat outputImage = IPCUDAK.sobelFilter(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "sobelFilter", "CUDAKernel", elapsedTimeMs
    , "");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::sobelFilterNPP(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorNPP IPNPP;
    cv::Mat outputImage = IPNPP.sobelFilter(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "sobelFilter", "NPP", elapsedTimeMs
        , "");

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::sobelFilterGStreamer(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // 시작 시간 측정

    ImageProcessorGStreamer IPGStreamer;
    cv::Mat outputImage = IPGStreamer.sobelFilter(inputImage);

    double endTime = cv::getTickCount(); // 종료 시간 측정
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

    result = setResult(result, inputImage, outputImage, "sobelFilter", "GStreamer", elapsedTimeMs
        , "");

    return result;
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
        redoStackNPP.push(lastProcessedImageNPP.clone());
        redoStackGStreamer.push(lastProcessedImageGStreamer.clone());

        // undo 스택에서 이미지 복원
        lastProcessedImageOpenCV = undoStackOpenCV.top().clone();
        lastProcessedImageIPP = undoStackIPP.top().clone();
        lastProcessedImageCUDA = undoStackCUDA.top().clone();
        lastProcessedImageCUDAKernel = undoStackCUDAKernel.top().clone();
        lastProcessedImageNPP = undoStackNPP.top().clone();
        lastProcessedImageGStreamer = undoStackGStreamer.top().clone();

        // undo 스택에서 이미지 제거
        undoStackOpenCV.pop();
        undoStackIPP.pop();
        undoStackCUDA.pop();
        undoStackCUDAKernel.pop();
        undoStackNPP.pop();
        undoStackGStreamer.pop();

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
        QString outputInfoNPP = "(Output) Channels: " + QString::number(lastProcessedImageNPP.channels())
            + ", type: " + QString::number(lastProcessedImageNPP.type())
            + ", depth: " + QString::number(lastProcessedImageNPP.depth());
        QString outputInfoGStreamer = "(Output) Channels: " + QString::number(lastProcessedImageGStreamer.channels())
            + ", type: " + QString::number(lastProcessedImageGStreamer.type())
            + ", depth: " + QString::number(lastProcessedImageGStreamer.depth());

        double endTime = cv::getTickCount(); // 종료 시간 측정
        double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

        // 결과 생성
        results.append(ProcessingResult(functionName, "OpenCV", lastProcessedImageOpenCV.clone(), elapsedTimeMs, "", outputInfoOpenCV));
        results.append(ProcessingResult(functionName, "IPP", lastProcessedImageIPP.clone(), elapsedTimeMs, "", outputInfoIPP));
        results.append(ProcessingResult(functionName, "CUDA", lastProcessedImageCUDA.clone(), elapsedTimeMs, "", outputInfoCUDA));
        results.append(ProcessingResult(functionName, "CUDAKernel", lastProcessedImageCUDAKernel.clone(), elapsedTimeMs, "", outputInfoCUDAKernel));
        results.append(ProcessingResult(functionName, "NPP", lastProcessedImageNPP.clone(), elapsedTimeMs, "", outputInfoNPP));
        results.append(ProcessingResult(functionName, "GStreamer", lastProcessedImageGStreamer.clone(), elapsedTimeMs, "", outputInfoGStreamer));

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
        undoStackNPP.push(lastProcessedImageNPP.clone());
        undoStackGStreamer.push(lastProcessedImageGStreamer.clone());

        // redo 스택에서 이미지 복원
        lastProcessedImageOpenCV = redoStackOpenCV.top().clone();
        lastProcessedImageIPP = redoStackIPP.top().clone();
        lastProcessedImageCUDA = redoStackCUDA.top().clone();
        lastProcessedImageCUDAKernel = redoStackCUDAKernel.top().clone();
        lastProcessedImageNPP = redoStackNPP.top().clone();
        lastProcessedImageGStreamer = redoStackGStreamer.top().clone();

        // redo 스택에서 이미지 제거
        redoStackOpenCV.pop();
        redoStackIPP.pop();
        redoStackCUDA.pop();
        redoStackCUDAKernel.pop();
        redoStackNPP.pop();
        redoStackGStreamer.pop();

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
        QString outputInfoNPP = "(Output) Channels: " + QString::number(lastProcessedImageNPP.channels())
            + ", type: " + QString::number(lastProcessedImageNPP.type())
            + ", depth: " + QString::number(lastProcessedImageNPP.depth());
        QString outputInfoGStreamer = "(Output) Channels: " + QString::number(lastProcessedImageGStreamer.channels())
            + ", type: " + QString::number(lastProcessedImageGStreamer.type())
            + ", depth: " + QString::number(lastProcessedImageGStreamer.depth());

        double endTime = cv::getTickCount(); // 종료 시간 측정
        double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // 시간 계산

        // 결과 생성
        results.append(ProcessingResult(functionName, "OpenCV", lastProcessedImageOpenCV.clone(), elapsedTimeMs, "", outputInfoOpenCV));
        results.append(ProcessingResult(functionName, "IPP", lastProcessedImageIPP.clone(), elapsedTimeMs, "", outputInfoIPP));
        results.append(ProcessingResult(functionName, "CUDA", lastProcessedImageCUDA.clone(), elapsedTimeMs, "", outputInfoCUDA));
        results.append(ProcessingResult(functionName, "CUDAKernel", lastProcessedImageCUDAKernel.clone(), elapsedTimeMs, "", outputInfoCUDAKernel));
        results.append(ProcessingResult(functionName, "NPP", lastProcessedImageNPP.clone(), elapsedTimeMs, "", outputInfoNPP));
        results.append(ProcessingResult(functionName, "GStreamer", lastProcessedImageGStreamer.clone(), elapsedTimeMs, "", outputInfoGStreamer));
        
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

    while (!undoStackNPP.empty()) {
        undoStackNPP.pop();
    }

    while (!undoStackGStreamer.empty()) {
        undoStackGStreamer.pop();
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

    while (!redoStackNPP.empty()) {
        redoStackNPP.pop();
    }

    while (!redoStackGStreamer.empty()) {
        redoStackGStreamer.pop();
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

const cv::Mat& ImageProcessor::getLastProcessedImageNPP() const
{
    return lastProcessedImageNPP;
}

const cv::Mat& ImageProcessor::getLastProcessedImageGStreamer() const
{
    return lastProcessedImageGStreamer;
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

void ImageProcessor::pushToUndoStackNPP(const cv::Mat& image)
{
    undoStackNPP.push(image.clone());
}

void ImageProcessor::pushToUndoStackGStreamer(const cv::Mat& image)
{
    undoStackGStreamer.push(image.clone());
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

void ImageProcessor::pushToRedoStackNPP(const cv::Mat& image)
{
    redoStackNPP.push(image.clone());
}

void ImageProcessor::pushToRedoStackGStreamer(const cv::Mat& image)
{
    redoStackGStreamer.push(image.clone());
}

ImageProcessor::ProcessingResult ImageProcessor::setResult(ProcessingResult& result
    , cv::Mat& inputImage, cv::Mat& outputImage, QString functionName
    , QString processName, double processingTime, QString argInfo)
{
    result.functionName = functionName;
    result.processName = processName;
    result.inputInfo = "\n(Input) Channels: " + QString::number(inputImage.channels())
        + ", type: " + QString::number(inputImage.type())
        + "(" + getImageTypeString(inputImage.type()) + ")"
        + ", depth: " + QString::number(inputImage.depth());
    result.processedImage = outputImage.clone();
    result.processingTime = processingTime;
    result.outputInfo = "\n(Output) Channels: " + QString::number(outputImage.channels())
        + ", type: " + QString::number(outputImage.type())
        + "(" + getImageTypeString(outputImage.type()) + ")"
        + ", depth: " + QString::number(outputImage.depth());
    result.argInfo = argInfo;

    return result;
}
