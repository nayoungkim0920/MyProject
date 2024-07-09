//ImageProcessor.h
#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

//����
//�ý��� ��� ����
//���̺귯�� ��� ����
//����� ���� ��� ����

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

#include <ipp.h>
#include <ipp/ippcore.h>
#include <ipp/ippi.h>
#include <ipp/ippcc.h>
#include <ipp/ipps.h>
#include "imageProcessing.cuh"


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

        ProcessingResult() = default;
        ProcessingResult(const QString& functionName, const QString& processName, const cv::Mat& processedImage, double processingTime)
            : functionName(functionName), processName(processName), processedImage(processedImage), processingTime(processingTime) {}
    };

    bool openImage(const std::string& fileName, cv::Mat& image);
    bool saveImage(const std::string& fileName, const cv::Mat& image);

    QFuture<bool> rotateImage(cv::Mat& image);
    QFuture<bool> zoominImage(cv::Mat& imageOpenCV, cv::Mat& imageIPP, cv::Mat& imageCUDA, cv::Mat& imageCUDAKernel, double scaleFactor);
    QFuture<bool> zoomoutImage(cv::Mat& image, double scaleFactor);
    QFuture<bool> grayScale(cv::Mat& imageOpenCV, cv::Mat& imageIPP, cv::Mat& imageCUDA, cv::Mat& imageCUDAKernel);
    QFuture<bool> gaussianBlur(cv::Mat& image, int kernelSize);
    QFuture<bool> cannyEdges(cv::Mat& image);
    QFuture<bool> medianFilter(cv::Mat& image);
    QFuture<bool> laplacianFilter(cv::Mat& image);
    QFuture<bool> bilateralFilter(cv::Mat& image);
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

signals: //�̺�Ʈ �߻��� �˸�
    void imageProcessed(QVector<ImageProcessor::ProcessingResult> results);
//slots: //�̺�Ʈ�� ó���ϴ� �Լ� ��Ī

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

    //bool grayScaleCUDA(cv::Mat& image);
    ProcessingResult grayScaleOpenCV(cv::Mat& image);
    ProcessingResult grayScaleIPP(cv::Mat& image);
    ProcessingResult grayScaleCUDA(cv::Mat& image);
    ProcessingResult grayScaleCUDAKernel(cv::Mat& image);

    ProcessingResult zoomInOpenCV(cv::Mat& image, double newWidth, double newHeight);
    ProcessingResult zoomInIPP(cv::Mat& image, double newWidth, double newHeight);
    ProcessingResult zoomInCUDA(cv::Mat& image, double newWidth, double newHeight);
    ProcessingResult zoomInCUDAKernel(cv::Mat& image, double newWidth, double newHeight);

    double getCurrentTimeMs();
};

#endif // IMAGEPROCESSOR_H