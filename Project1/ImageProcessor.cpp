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

QFuture<bool> ImageProcessor::rotateImage(cv::Mat& image) {

    return QtConcurrent::run([this, &image]()->bool {

        QMutexLocker locker(&mutex);

        try {

            if (image.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            //이미지 회전
            cv::Mat rotatedImage;
            cv::rotate(image, rotatedImage, cv::ROTATE_90_CLOCKWISE);

            image = rotatedImage.clone();
            lastProcessedImage = image.clone();

            emit imageProcessed(image);

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
    return QtConcurrent::run([this, &image, scaleFactor]() -> bool {
        try {
            if (image.empty()) {
                qDebug() << "입력 이미지가 비어 있습니다.";
                return false;
            }

            if (scaleFactor <= 0) {
                qDebug() << "잘못된 확대/축소 배율입니다.";
                return false;
            }

            QMutexLocker locker(&mutex);

            int newWidth = static_cast<int>(image.cols * scaleFactor);
            int newHeight = static_cast<int>(image.rows * scaleFactor);

            cv::Mat zoomedImage;
            cv::resize(image, zoomedImage, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
            image = zoomedImage.clone(); // 이미지를 복사하여 업데이트

            emit imageProcessed(image); // 이미지 처리 완료 시그널 발생

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "이미지 확대/축소 중 예외가 발생했습니다:" << e.what();
            return false;
        }
        });
}

QFuture<bool> ImageProcessor::convertToGrayscale(cv::Mat& image)
{
    return QtConcurrent::run([this, &image]() -> bool {
        QMutexLocker locker(&mutex);
        try {
            if (image.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            cv::Mat grayImage;

            // BGR 이미지를 RGB 채널로 분리
            std::vector<cv::Mat> channels;
            cv::split(image, channels);

            // RGB 채널을 그레이스케일로 변환
            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

            // 그레이스케일 이미지를 RGB 형식으로 변환
            cv::Mat merged;
            cv::merge(std::vector<cv::Mat>{gray, gray, gray}, merged);

            // 원본 이미지에 그레이스케일 이미지 적용
            image = merged.clone();
            lastProcessedImage = image.clone();

            emit imageProcessed(image); // 이미지 처리가 완료되었음을 시그널로 알림
            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "Exception occurred while converting to grayscale:" << e.what();
            return false;
        }
        });
}

QFuture<bool> ImageProcessor::applyGaussianBlur(cv::Mat& image, int kernelSize)
{
    return QtConcurrent::run([this, &image, kernelSize]() -> bool {
        QMutexLocker locker(&mutex);
        try {
            if (image.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            if (kernelSize % 2 == 0 || kernelSize < 1) {
                qDebug() << "Invalid kernel size for Gaussian blur.";
                return false;
            }

            cv::Mat blurredImage;
            cv::GaussianBlur(image, blurredImage, cv::Size(kernelSize, kernelSize), 0, 0);

            image = blurredImage.clone();
            lastProcessedImage = image.clone();

            emit imageProcessed(image);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "Exception occurred while applying Gaussian blur:"
                << e.what();
            return false;
        }
    });
}

QFuture<bool> ImageProcessor::detectEdges(cv::Mat& image)
{
    return QtConcurrent::run([this, &image]() -> bool {
        QMutexLocker locker(&mutex);
        try {
            if (image.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            cv::Mat grayImage;
            cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

            cv::Mat edges;
            cv::Canny(grayImage, edges, 50, 150);

            cv::Mat outputImage;
            image.copyTo(outputImage);
            outputImage.setTo(cv::Scalar(0, 255, 0), edges); // Green edges

            image = outputImage.clone();
            lastProcessedImage = image.clone();

            emit imageProcessed(image);
            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "Exception occurred while detecting edges:" << e.what();
            return false;
        }
        });
}
