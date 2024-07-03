//ImageProcessor.h
#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include <stack>
#include <QObject>
#include <QDebug>
#include <chrono>
#include <QMutex>
#include <QMutexLocker>
#include <QtConcurrent/QtConcurrent>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <ipp.h>
//#include <ipp/ippi.h>
//#include <ipp/ippcc.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include "imageProcessing.cuh"

class ImageProcessor : public QObject
{
    Q_OBJECT

public:
    explicit ImageProcessor(QObject* parent = nullptr);
    ~ImageProcessor();

    bool openImage(const std::string& fileName, cv::Mat& image);
    bool saveImage(const std::string& fileName, const cv::Mat& image);
    QFuture<bool> rotateImage(cv::Mat& image);
    QFuture<bool> zoominImage(cv::Mat& image, double scaleFactor);
    QFuture<bool> zoomoutImage(cv::Mat& image, double scaleFactor);
    QFuture<bool> grayScale(cv::Mat& image);
    QFuture<bool> gaussianBlur(cv::Mat& image, int kernelSize);
    QFuture<bool> cannyEdges(cv::Mat& image);
    QFuture<bool> medianFilter(cv::Mat& image);
    QFuture<bool> laplacianFilter(cv::Mat& image);
    QFuture<bool> bilateralFilter(cv::Mat& image);
    QFuture<bool> sobelFilter(cv::Mat& image);

    bool canUndo() const;
    bool canRedo() const;
    void undo();
    void redo();
    void cleanUndoStack();
    void cleanRedoStack();
    void initializeCUDA();

    const cv::Mat& getLastProcessedImage() const;

signals: //이벤트 발생을 알림
    void imageProcessed(const cv::Mat& processedImage, double processingTimeMs, QString processName);
//slots: //이벤트를 처리하는 함수 지칭

private:
    cv::Mat lastProcessedImage;
    QMutex mutex;
    std::stack<cv::Mat> undoStack;
    std::stack<cv::Mat> redoStack;

    void pushToUndoStack(const cv::Mat& image);
    void pushToRedoStack(const cv::Mat& image);

    bool grayScaleCUDA(cv::Mat& image);

    double getCurrentTimeMs();
};

#endif // IMAGEPROCESSOR_H