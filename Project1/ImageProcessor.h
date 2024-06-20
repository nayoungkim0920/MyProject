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

signals: //�̺�Ʈ �߻��� �˸�
    void imageProcessed(const cv::Mat& processedImage);

//slots: //�̺�Ʈ�� ó���ϴ� �Լ� ��Ī

private: 
    cv::Mat lastProcessedImage;
    QMutex mutex;
};

#endif // IMAGEPROCESSOR_H
