//ImageProcessor.h
#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

//순서
//시스템 헤더 파일
//라이브러리 헤더 파일
//사용자 정의 헤더 파일

#include <QObject>
#include <QDebug>
#include <chrono>
#include <stack>
#include <vector>
#include <QMutex>
#include <QMutexLocker>
#include <QtConcurrent/QtConcurrent>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>

#include <omp.h>

#include <npp.h>
#include <nppi.h>
#include <nppi_filtering_functions.h>

#include <ipp.h>
#include <ipp/ippcore.h>
#include <ipp/ippi.h>
#include <ipp/ippcc.h>
#include <ipp/ipps.h>
#include <ipp/ippcv.h>
#include "imageProcessing.cuh"
#include "ImageTypeConverter.h"

#ifndef MAX_NUM_THREADS
#define MAX_NUM_THREADS 8 // 예시로 임의로 설정
#endif

class ImageProcessor : public QObject
{
    Q_OBJECT

public:
    explicit ImageProcessor(QObject* parent = nullptr);
    ~ImageProcessor();

    struct ProcessingResult {
        QString functionName;
        QString processName;
        cv::Mat processedImage;
        double processingTime;
        QString inputInfo;
        QString outputInfo;

        ProcessingResult() = default;
        ProcessingResult(const QString& functionName, const QString& processName, const cv::Mat& processedImage, double processingTime, const QString& inputInfo, const QString& outputInfo)
            : functionName(functionName), processName(processName), processedImage(processedImage), processingTime(processingTime), inputInfo(inputInfo), outputInfo(outputInfo) {}
    };

    bool openImage(const std::string& fileName, cv::Mat& image);
    bool saveImage(const std::string& fileName, const cv::Mat& image);

    QFuture<bool> rotateImage(cv::Mat& imageOpenCV, cv::Mat& imageIPP, cv::Mat& imageCUDA, cv::Mat& imageCUDAKernel);
    QFuture<bool> zoomInImage(cv::Mat& imageOpenCV, cv::Mat& imageIPP, cv::Mat& imageCUDA, cv::Mat& imageCUDAKernel, double scaleFactor);
    QFuture<bool> zoomOutImage(cv::Mat& imageOpenCV, cv::Mat& imageIPP, cv::Mat& imageCUDA, cv::Mat& imageCUDAKernel, double scaleFactor);
    QFuture<bool> grayScale(cv::Mat& imageOpenCV, cv::Mat& imageIPP, cv::Mat& imageCUDA, cv::Mat& imageCUDAKernel);
    QFuture<bool> gaussianBlur(cv::Mat& imageOpenCV, cv::Mat& imageIPP, cv::Mat& imageCUDA, cv::Mat& imageCUDAKernel, int kernelSize);
    QFuture<bool> cannyEdges(cv::Mat& imageOpenCV, cv::Mat& imageIPP, cv::Mat& imageCUDA, cv::Mat& imageCUDAKernel);
    QFuture<bool> medianFilter(cv::Mat& imageOpenCV, cv::Mat& imageIPP, cv::Mat& imageCUDA, cv::Mat& imageCUDAKernel);
    QFuture<bool> laplacianFilter(cv::Mat& imageOpenCV, cv::Mat& imageIPP, cv::Mat& imageCUDA, cv::Mat& imageCUDAKernel);
    QFuture<bool> bilateralFilter(cv::Mat& imageOpenCV, cv::Mat& imageIPP, cv::Mat& imageCUDA, cv::Mat& imageCUDAKernel);
    QFuture<bool> sobelFilter(cv::Mat& image);

    bool canUndoOpenCV() const;
    bool canRedoOpenCV() const;

    void undo();
    void redo();

    void cleanUndoStack();
    void cleanRedoStack();

    void initializeCUDA();

    const cv::Mat& getLastProcessedImageOpenCV() const;
    const cv::Mat& getLastProcessedImageIPP() const;
    const cv::Mat& getLastProcessedImageCUDA() const;
    const cv::Mat& getLastProcessedImageCUDAKernel() const;

signals: //이벤트 발생을 알림
    void imageProcessed(QVector<ImageProcessor::ProcessingResult> results);
//slots: //이벤트를 처리하는 함수 지칭

private:

    cv::Mat lastProcessedImageOpenCV;
    cv::Mat lastProcessedImageIPP;
    cv::Mat lastProcessedImageCUDA;
    cv::Mat lastProcessedImageCUDAKernel;

    QMutex mutex;

    std::stack<cv::Mat> undoStackOpenCV;
    std::stack<cv::Mat> undoStackIPP;
    std::stack<cv::Mat> undoStackCUDA;
    std::stack<cv::Mat> undoStackCUDAKernel;

    std::stack<cv::Mat> redoStackOpenCV;
    std::stack<cv::Mat> redoStackIPP;
    std::stack<cv::Mat> redoStackCUDA;
    std::stack<cv::Mat> redoStackCUDAKernel;

    void pushToUndoStackOpenCV(const cv::Mat& image);
    void pushToUndoStackIPP(const cv::Mat& image);
    void pushToUndoStackCUDA(const cv::Mat& image);
    void pushToUndoStackCUDAKernel(const cv::Mat& image);

    void pushToRedoStackOpenCV(const cv::Mat& image);
    void pushToRedoStackIPP(const cv::Mat& image);
    void pushToRedoStackCUDA(const cv::Mat& image);
    void pushToRedoStackCUDAKernel(const cv::Mat& image);

    ProcessingResult setResult(ProcessingResult& result, cv::Mat& inputImage
        , cv::Mat& outputImage, QString functionName, QString processName, double processingTime);

    //bool grayScaleCUDA(cv::Mat& image);

    ProcessingResult grayScaleOpenCV(cv::Mat& inputImage);
    ProcessingResult grayScaleIPP(cv::Mat& inputImage);
    ProcessingResult grayScaleCUDA(cv::Mat& inputImage);
    ProcessingResult grayScaleCUDAKernel(cv::Mat& inputImage);

    ProcessingResult zoomOpenCV(cv::Mat& inputImage, double newWidth, double newHeight);
    ProcessingResult zoomIPP(cv::Mat& inputImage, double newWidth, double newHeight);
    ProcessingResult zoomCUDA(cv::Mat& inputImage, double newWidth, double newHeight);
    ProcessingResult zoomCUDAKernel(cv::Mat& inputImage, double newWidth, double newHeight);

    ProcessingResult rotateOpenCV(cv::Mat& inputImage);
    ProcessingResult rotateIPP(cv::Mat& inputImage);
    ProcessingResult rotateCUDA(cv::Mat& inputImage);
    ProcessingResult rotateCUDAKernel(cv::Mat& inputImage);

    ProcessingResult gaussianBlurOpenCV(cv::Mat& inputImage, int kernelSize);
    ProcessingResult gaussianBlurIPP(cv::Mat& inputImage, int kernelSize);
    ProcessingResult gaussianBlurCUDA(cv::Mat& inputImage, int kernelSize);
    ProcessingResult gaussianBlurCUDAKernel(cv::Mat& inputImage, int kernelSize);

    ProcessingResult cannyEdgesOpenCV(cv::Mat& inputImage);
    ProcessingResult cannyEdgesIPP(cv::Mat& inputImage);
    ProcessingResult cannyEdgesCUDA(cv::Mat& inputImage);
    ProcessingResult cannyEdgesCUDAKernel(cv::Mat& inputImage);

    cv::Mat convertToGrayOpenCV(const cv::Mat& inputImage);
    cv::Mat convertToGrayIPP(const cv::Mat& inputImage);
    cv::Mat convertToGrayCUDA(const cv::Mat& inputImage);
    cv::Mat convertToGrayCUDAKernel(cv::Mat& inputImage);

    ProcessingResult medianFilterOpenCV(cv::Mat& inputImage);
    ProcessingResult medianFilterIPP(cv::Mat& inputImage);
    ProcessingResult medianFilterCUDA(cv::Mat& inputImage);
    ProcessingResult medianFilterCUDAKernel(cv::Mat& inputImage);

    ProcessingResult laplacianFilterOpenCV(cv::Mat& inputImage);
    ProcessingResult laplacianFilterIPP(cv::Mat& inputImage);
    ProcessingResult laplacianFilterCUDA(cv::Mat& inputImage);
    ProcessingResult laplacianFilterCUDAKernel(cv::Mat& inputImage);

    ProcessingResult bilateralFilterOpenCV(cv::Mat& inputImage);
    ProcessingResult bilateralFilterIPP(cv::Mat& inputImage);
    ProcessingResult bilateralFilterCUDA(cv::Mat& inputImage);
    ProcessingResult bilateralFilterCUDAKernel(cv::Mat& inputImage);

    double getCurrentTimeMs();
    // 유틸리티 함수 선언
    Ipp8u* matToIpp8u(cv::Mat& mat);
};

#endif // IMAGEPROCESSOR_H