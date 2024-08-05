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

#include "ImageTypeConverter.h"

#include "ImageProcessorOpenCV.h"
#include "ImageProcessorIPP.h"
#include "ImageProcessorCUDA.h"
#include "ImageProcessorCUDAKernel.h"
#include "ImageProcessorNPP.h"
#include "ImageProcessorGStreamer.h"

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
        QString argInfo;

        ProcessingResult() = default;
        ProcessingResult(const QString& functionName, const QString& processName, const cv::Mat& processedImage, double processingTime, const QString& inputInfo, const QString& outputInfo, const QString& argInfo = "")
            : functionName(functionName), processName(processName), processedImage(processedImage), processingTime(processingTime), inputInfo(inputInfo), outputInfo(outputInfo), argInfo(argInfo) {}
    };

    bool openImage(const std::string& fileName, cv::Mat& image);
    bool saveImage(const std::string& fileName, const cv::Mat& image);

    QFuture<bool> rotateImage(cv::Mat& imageOpenCV, cv::Mat& imageIPP, cv::Mat& imageCUDA, cv::Mat& imageCUDAKernel, cv::Mat& imageNPP, cv::Mat& imageGStreamer);
    QFuture<bool> zoomInImage(cv::Mat& imageOpenCV, cv::Mat& imageIPP, cv::Mat& imageCUDA, cv::Mat& imageCUDAKernel, cv::Mat& imageNPP, cv::Mat& imageGStreamer, double scaleFactor);
    QFuture<bool> zoomOutImage(cv::Mat& imageOpenCV, cv::Mat& imageIPP, cv::Mat& imageCUDA, cv::Mat& imageCUDAKernel, cv::Mat& imageNPP, cv::Mat& imageGStreamer, double scaleFactor);
    QFuture<bool> grayScale(cv::Mat& imageOpenCV, cv::Mat& imageIPP, cv::Mat& imageCUDA, cv::Mat& imageCUDAKernel, cv::Mat& imageNPP, cv::Mat& imageGStreamer);
    QFuture<bool> gaussianBlur(cv::Mat& imageOpenCV, cv::Mat& imageIPP, cv::Mat& imageCUDA, cv::Mat& imageCUDAKernel, cv::Mat& NPP, cv::Mat& imageGStreamer, int kernelSize);
    QFuture<bool> cannyEdges(cv::Mat& imageOpenCV, cv::Mat& imageIPP, cv::Mat& imageCUDA, cv::Mat& imageCUDAKernel, cv::Mat& imageNPP, cv::Mat& imageGStreamer);
    QFuture<bool> medianFilter(cv::Mat& imageOpenCV, cv::Mat& imageIPP, cv::Mat& imageCUDA, cv::Mat& imageCUDAKernel, cv::Mat& imageNPP, cv::Mat& imageGStreamer);
    QFuture<bool> laplacianFilter(cv::Mat& imageOpenCV, cv::Mat& imageIPP, cv::Mat& imageCUDA, cv::Mat& imageCUDAKernel, cv::Mat& imageNPP, cv::Mat& imageGStreamer);
    QFuture<bool> bilateralFilter(cv::Mat& imageOpenCV, cv::Mat& imageIPP, cv::Mat& imageCUDA, cv::Mat& imageCUDAKernel, cv::Mat& imageNPP, cv::Mat& imageGStreamer);
    QFuture<bool> sobelFilter(cv::Mat& imageOpenCV, cv::Mat& imageIPP, cv::Mat& imageCUDA, cv::Mat& imageCUDAKernel, cv::Mat& imageNPP, cv::Mat& imageGStreamer);

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
    const cv::Mat& getLastProcessedImageNPP() const;
    const cv::Mat& getLastProcessedImageGStreamer() const;

signals: //이벤트 발생을 알림
    void imageProcessed(QVector<ImageProcessor::ProcessingResult> results);
    //slots: //이벤트를 처리하는 함수 지칭

private:

    cv::Mat lastProcessedImageOpenCV;
    cv::Mat lastProcessedImageIPP;
    cv::Mat lastProcessedImageCUDA;
    cv::Mat lastProcessedImageCUDAKernel;
    cv::Mat lastProcessedImageNPP;
    cv::Mat lastProcessedImageGStreamer;

    QMutex mutex;

    std::stack<cv::Mat> undoStackOpenCV;
    std::stack<cv::Mat> undoStackIPP;
    std::stack<cv::Mat> undoStackCUDA;
    std::stack<cv::Mat> undoStackCUDAKernel;
    std::stack<cv::Mat> undoStackNPP;
    std::stack<cv::Mat> undoStackGStreamer;

    std::stack<cv::Mat> redoStackOpenCV;
    std::stack<cv::Mat> redoStackIPP;
    std::stack<cv::Mat> redoStackCUDA;
    std::stack<cv::Mat> redoStackCUDAKernel;
    std::stack<cv::Mat> redoStackNPP;
    std::stack<cv::Mat> redoStackGStreamer;

    void pushToUndoStackOpenCV(const cv::Mat& image);
    void pushToUndoStackIPP(const cv::Mat& image);
    void pushToUndoStackCUDA(const cv::Mat& image);
    void pushToUndoStackCUDAKernel(const cv::Mat& image);
    void pushToUndoStackNPP(const cv::Mat& image);
    void pushToUndoStackGStreamer(const cv::Mat& image);

    void pushToRedoStackOpenCV(const cv::Mat& image);
    void pushToRedoStackIPP(const cv::Mat& image);
    void pushToRedoStackCUDA(const cv::Mat& image);
    void pushToRedoStackCUDAKernel(const cv::Mat& image);
    void pushToRedoStackNPP(const cv::Mat& image);
    void pushToRedoStackGStreamer(const cv::Mat& image);

    ProcessingResult setResult(ProcessingResult& result, cv::Mat& inputImage
        , cv::Mat& outputImage, QString functionName, QString processName
        , double processingTime, QString arginfo);

    ProcessingResult grayScaleOpenCV(cv::Mat& inputImage);
    ProcessingResult grayScaleIPP(cv::Mat& inputImage);
    ProcessingResult grayScaleCUDA(cv::Mat& inputImage);
    ProcessingResult grayScaleCUDAKernel(cv::Mat& inputImage);
    ProcessingResult grayScaleNPP(cv::Mat& inputImage);
    ProcessingResult grayScaleGStreamer(cv::Mat& inputImage);

    ProcessingResult zoomOpenCV(cv::Mat& inputImage, double newWidth, double newHeight);
    ProcessingResult zoomIPP(cv::Mat& inputImage, double newWidth, double newHeight);
    ProcessingResult zoomCUDA(cv::Mat& inputImage, double newWidth, double newHeight);
    ProcessingResult zoomCUDAKernel(cv::Mat& inputImage, double newWidth, double newHeight);
    ProcessingResult zoomNPP(cv::Mat& inputImage, double newWidth, double newHeight);
    ProcessingResult zoomGStreamer(cv::Mat& inputImage, double newWidth, double newHeight);

    ProcessingResult rotateOpenCV(cv::Mat& inputImage);
    ProcessingResult rotateIPP(cv::Mat& inputImage);
    ProcessingResult rotateCUDA(cv::Mat& inputImage);
    ProcessingResult rotateCUDAKernel(cv::Mat& inputImage);
    ProcessingResult rotateNPP(cv::Mat& inputImage);
    ProcessingResult rotateGStreamer(cv::Mat& inputImage);

    ProcessingResult gaussianBlurOpenCV(cv::Mat& inputImage, int kernelSize);
    ProcessingResult gaussianBlurIPP(cv::Mat& inputImage, int kernelSize);
    ProcessingResult gaussianBlurCUDA(cv::Mat& inputImage, int kernelSize);
    ProcessingResult gaussianBlurCUDAKernel(cv::Mat& inputImage, int kernelSize);
    ProcessingResult gaussianBlurNPP(cv::Mat& inputImage, int kernelSize);
    ProcessingResult gaussianBlurGStreamer(cv::Mat& inputImage, int kernelSize);

    ProcessingResult cannyEdgesOpenCV(cv::Mat& inputImage);
    ProcessingResult cannyEdgesIPP(cv::Mat& inputImage);
    ProcessingResult cannyEdgesCUDA(cv::Mat& inputImage);
    ProcessingResult cannyEdgesCUDAKernel(cv::Mat& inputImage);
    ProcessingResult cannyEdgesNPP(cv::Mat& inputImage);
    ProcessingResult cannyEdgesGStreamer(cv::Mat& inputImage);

    ProcessingResult medianFilterOpenCV(cv::Mat& inputImage);
    ProcessingResult medianFilterIPP(cv::Mat& inputImage);
    ProcessingResult medianFilterCUDA(cv::Mat& inputImage);
    ProcessingResult medianFilterCUDAKernel(cv::Mat& inputImage);
    ProcessingResult medianFilterNPP(cv::Mat& inputImage);
    ProcessingResult medianFilterGStreamer(cv::Mat& inputImage);

    ProcessingResult laplacianFilterOpenCV(cv::Mat& inputImage);
    ProcessingResult laplacianFilterIPP(cv::Mat& inputImage);
    ProcessingResult laplacianFilterCUDA(cv::Mat& inputImage);
    ProcessingResult laplacianFilterCUDAKernel(cv::Mat& inputImage);
    ProcessingResult laplacianFilterNPP(cv::Mat& inputImage);
    ProcessingResult laplacianFilterGStreamer(cv::Mat& inputImage);

    ProcessingResult bilateralFilterOpenCV(cv::Mat& inputImage);
    ProcessingResult bilateralFilterIPP(cv::Mat& inputImage);
    ProcessingResult bilateralFilterCUDA(cv::Mat& inputImage);
    ProcessingResult bilateralFilterCUDAKernel(cv::Mat& inputImage);
    ProcessingResult bilateralFilterNPP(cv::Mat& inputImage);
    ProcessingResult bilateralFilterGStreamer(cv::Mat& inputImage);

    ProcessingResult sobelFilterOpenCV(cv::Mat& inputImage);
    ProcessingResult sobelFilterIPP(cv::Mat& inputImage);
    ProcessingResult sobelFilterCUDA(cv::Mat& inputImage);
    ProcessingResult sobelFilterCUDAKernel(cv::Mat& inputImage);
    ProcessingResult sobelFilterNPP(cv::Mat& inputImage);
    ProcessingResult sobelFilterGStreamer(cv::Mat& inputImage);

    double getCurrentTimeMs();
};

#endif // IMAGEPROCESSOR_H