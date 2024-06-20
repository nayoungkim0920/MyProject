//ImageProcessor.h
#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

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

signals: //이벤트 발생을 알림
    void imageProcessed(const cv::Mat& processedImage);

//slots: //이벤트를 처리하는 함수 지칭

private: 
    cv::Mat lastProcessedImage;
    QMutex mutex;
};

#endif // IMAGEPROCESSOR_H
