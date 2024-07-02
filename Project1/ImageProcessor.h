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
#include <opencv2/highgui.hpp>
//#include <ipp.h>
//#include <ipp/ippi.h>
//#include <ipp/ippcc.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

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
    QFuture<bool> cannyEdges(cv::Mat& image);
    QFuture<bool> medianFilter(cv::Mat& image);
    QFuture<bool> laplacianFilter(cv::Mat& image);
    QFuture<bool> bilateralFilter(cv::Mat& image);

    bool canUndo() const;
    bool canRedo() const;
    void undo();
    void redo();
    void cleanUndoStack();
    void cleanRedoStack();

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
<<<<<<< HEAD

    bool convertToGrayscaleCUDA(cv::Mat& image);
=======
>>>>>>> 1411ff1ca5f8ad193b2e19cf1fb730e230fbea1e
};

#endif // IMAGEPROCESSOR_H
