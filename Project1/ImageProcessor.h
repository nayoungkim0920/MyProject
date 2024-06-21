//ImageProcessor.h
#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include <stack>
#include <QObject>
#include <QDebug>
#include <QMutex>
#include <QMutexLocker>
#include <QtConcurrent/QtConcurrent>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
//#include <ipp.h>
//#include <ipp/ippi.h>
//#include <ipp/ippcc.h>
//#include <opencv2/cuda>

class ImageProcessor : public QObject
{
    Q_OBJECT

public:
    explicit ImageProcessor(QObject* parent = nullptr);
    ~ImageProcessor();

    bool openImage(const std::string& fileName, cv::Mat& image);
    bool saveImage(const std::string& fileName, const cv::Mat& image);
    QFuture<bool> rotateImage(cv::Mat& image);
    QFuture<bool> zoomImage(cv::Mat& image, double scaleFactor);
    QFuture<bool> convertToGrayscaleAsync(cv::Mat& image);
    QFuture<bool> applyGaussianBlur(cv::Mat& image, int kernelSize);
    QFuture<bool> detectEdges(cv::Mat& image);

    bool canUndo() const;
    bool canRedo() const;
    void undo();
    void redo();

    const cv::Mat& getLastProcessedImage() const;

signals: //이벤트 발생을 알림
    void imageProcessed(const cv::Mat& processedImage);

//slots: //이벤트를 처리하는 함수 지칭

private: 
    cv::Mat lastProcessedImage;
    QMutex mutex;
    std::stack<cv::Mat> undoStack;
    std::stack<cv::Mat> redoStack;

    void pushToUndoStack(const cv::Mat& image);
    void pushToRedoStack(const cv::Mat& image);

    cv::Mat converToGrayScale(const cv::Mat& image);
};

#endif // IMAGEPROCESSOR_H
