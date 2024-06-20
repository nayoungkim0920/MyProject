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

QFuture<bool> ImageProcessor::rotateImage(cv::Mat& image)
{
    return QtConcurrent::run([this, &image]() -> bool {
        try {
            if (image.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            QMutexLocker locker(&mutex);

            cv::Mat rotatedImage;
            cv::transpose(image, rotatedImage);
            cv::flip(rotatedImage, rotatedImage, 1);
            image = rotatedImage;
            lastProcessedImage = image.clone();//บนป็

            emit imageProcessed(lastProcessedImage);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "Exception occured while rotating image:"
                << e.what();
            return false;
        }
    });
}

QFuture<bool> ImageProcessor::zoomImage(cv::Mat& image, double scaleFactor)
{
    return QtConcurrent::run([this, &image, scaleFactor]()->bool {
        try {
            if (image.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            if (scaleFactor <= 0) {
                qDebug() << "Invalid scale factor.";
                return false;
            }

            QMutexLocker locker(&mutex);

            int newWidth = static_cast<int>(image.cols * scaleFactor);
            int newHeight = static_cast<int>(image.rows * scaleFactor);

            cv::Mat zoomedImage;
            cv::resize(image,
                zoomedImage,
                cv::Size(newWidth, newHeight),
                0,0,cv::INTER_LINEAR);
            image = zoomedImage;
            lastProcessedImage = image.clone();

            emit imageProcessed(lastProcessedImage);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "Exception occurred while zooming image:"
                << e.what();
            return false;
        }
    });
}