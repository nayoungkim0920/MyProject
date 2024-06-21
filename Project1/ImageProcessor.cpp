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

            //�̹��� ȸ��
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
                qDebug() << "�Է� �̹����� ��� �ֽ��ϴ�.";
                return false;
            }

            if (scaleFactor <= 0) {
                qDebug() << "�߸��� Ȯ��/��� �����Դϴ�.";
                return false;
            }

            QMutexLocker locker(&mutex);

            int newWidth = static_cast<int>(image.cols * scaleFactor);
            int newHeight = static_cast<int>(image.rows * scaleFactor);

            cv::Mat zoomedImage;
            cv::resize(image, zoomedImage, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
            image = zoomedImage.clone(); // �̹����� �����Ͽ� ������Ʈ

            emit imageProcessed(image); // �̹��� ó�� �Ϸ� �ñ׳� �߻�

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "�̹��� Ȯ��/��� �� ���ܰ� �߻��߽��ϴ�:" << e.what();
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

            // BGR �̹����� RGB ä�η� �и�
            std::vector<cv::Mat> channels;
            cv::split(image, channels);

            // RGB ä���� �׷��̽����Ϸ� ��ȯ
            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

            // �׷��̽����� �̹����� RGB �������� ��ȯ
            cv::Mat merged;
            cv::merge(std::vector<cv::Mat>{gray, gray, gray}, merged);

            // ���� �̹����� �׷��̽����� �̹��� ����
            image = merged.clone();
            lastProcessedImage = image.clone();

            emit imageProcessed(image); // �̹��� ó���� �Ϸ�Ǿ����� �ñ׳η� �˸�
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
