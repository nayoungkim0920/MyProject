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

            pushToUndoStack(image);

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
    return QtConcurrent::run([this, &image, scaleFactor]() -> bool {
        
        QMutexLocker locker(&mutex);        
        
        try {

            pushToUndoStack(image);
            
            if (image.empty()) {
                qDebug() << "�Է� �̹����� ��� �ֽ��ϴ�.";
                return false;
            }

            if (scaleFactor <= 0) {
                qDebug() << "�߸��� Ȯ��/��� �����Դϴ�.";
                return false;
            }            

            int newWidth = static_cast<int>(image.cols * scaleFactor);
            int newHeight = static_cast<int>(image.rows * scaleFactor);

            cv::Mat zoomedImage;
            cv::resize(image, zoomedImage, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
            
            image = zoomedImage.clone(); // �̹����� �����Ͽ� ������Ʈ
            lastProcessedImage = image.clone();

            emit imageProcessed(image); // �̹��� ó�� �Ϸ� �ñ׳� �߻�

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "�̹��� Ȯ��/��� �� ���ܰ� �߻��߽��ϴ�:" << e.what();
            return false;
        }
        });
}

QFuture<bool> ImageProcessor::convertToGrayscaleAsync(cv::Mat& image)
{
    return QtConcurrent::run([this, &image]() -> bool {

        QMutexLocker locker(&mutex);        

        try {

            pushToUndoStack(image);

            if (image.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            cv::Mat gray = convertToGrayScale(image);

            // ���� �̹����� �׷��̽����� �̹��� ����
            image = gray.clone();
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

            pushToUndoStack(image);

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

//Canny
QFuture<bool> ImageProcessor::cannyEdges(cv::Mat& image)
{
    return QtConcurrent::run([this, &image]() -> bool {

        QMutexLocker locker(&mutex);        

        try {

            pushToUndoStack(image);

            if (image.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            //ȸ�����̹���
            cv::Mat gray = convertToGrayScale(image);

            cv::Mat edges;//�����������
            //50:���� �Ӱ谪(threshold1)
            //�����ĺ��� ����Ǳ� ���� �ּ� ���� 
            //�Ӱ谪���� ���� ������ ������ ���ŵȴ�.
            //150:���� �Ӱ谪(threshold2)
            //���� ������ ���ֵǴ� �Ӱ谪
            //�� �Ӱ谪 ������ ������ ���� ������ ����Ǿ�
            //�ϳ��� ������ �����ȴ�
            cv::Canny(gray, edges, 50, 150);

            cv::Mat outputImage;
            gray.copyTo(outputImage);
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

QFuture<bool> ImageProcessor::medianFilter(cv::Mat& image)
{
    return QtConcurrent::run([this, &image]()->bool {

        QMutexLocker locker(&mutex);        

        try {

            pushToUndoStack(image);

            if (image.empty()) {
                qDebug() << "median ���͸� ������ �̹����� �����ϴ�.";
                return false;
            }

            cv::Mat medianedImage;
            cv::medianBlur(image, medianedImage, 5);
            image = medianedImage.clone();
            lastProcessedImage = image.clone();

            emit imageProcessed(image);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "median ���� ���� �� ���� �߻�: "
                << e.what();
            return false;
        }
    });
}

QFuture<bool> ImageProcessor::laplacianFilter(cv::Mat& image)
{
    return QtConcurrent::run([this, &image]()->bool {

        QMutexLocker locker(&mutex);        

        try {

            pushToUndoStack(image);

            if (image.empty()) {
                qDebug() << "laplacian ���͸� ������ �̹����� �����ϴ�.";
                return false;
            }

            cv::Mat filteredImage;
            cv::Laplacian(image, filteredImage, CV_8U, 3);

            image = filteredImage.clone();
            lastProcessedImage = image.clone();

            emit imageProcessed(image);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "laplacian ���� ���� �� ���� �߻�: "
                << e.what();
            return false;
        }
        });
}

QFuture<bool> ImageProcessor::bilateralFilter(cv::Mat& image)
{
    return QtConcurrent::run([this, &image]()->bool {

        QMutexLocker locker(&mutex);        

        try {

            pushToUndoStack(image);

            if (image.empty()) {
                qDebug() << "bilateral ���͸� ������ �̹����� �����ϴ�.";
                return false;
            }

            cv::Mat filteredImage;
            cv::bilateralFilter(image, filteredImage, 9, 75, 75);

            image = filteredImage.clone();
            lastProcessedImage = image.clone();

            emit imageProcessed(image);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "bilateral ���� ���� �� ���� �߻�: "
                << e.what();
            return false;
        }
        });
}

bool ImageProcessor::canUndo() const
{
    return !undoStack.empty();
}

bool ImageProcessor::canRedo() const
{
    return !redoStack.empty();
}

//�������
void ImageProcessor::undo()
{
    try {

        if (!canUndo())
            throw std::runtime_error("Cannot redo: undo stack is empty");

        cv::Mat imageToRestore = undoStack.top();
        redoStack.push(lastProcessedImage);
        lastProcessedImage = imageToRestore.clone();
        undoStack.pop();

        emit imageProcessed(lastProcessedImage);

    }
    catch (const std::exception& e) {
        qDebug() << "Exception occurred in ImageProcessor::undo(): "
            << e.what();
    }
}

//�����
void ImageProcessor::redo()
{
    try {

        if (!canRedo())
            throw std::runtime_error("Cannot redo: redo stack is empty");

        cv::Mat imageToRestore = redoStack.top();
        undoStack.push(lastProcessedImage);
        lastProcessedImage = imageToRestore.clone();
        redoStack.pop();

        emit imageProcessed(lastProcessedImage);

    }
    catch (const std::exception& e) {
        qDebug() << "Exception occurred in ImageProcessor::redo(): " 
            << e.what();
    }
}

void ImageProcessor::cleanUndoStack()
{
    QMutexLocker locker(&mutex);
    while (!undoStack.empty()) {
        undoStack.pop();
    }
}

void ImageProcessor::cleanRedoStack()
{
    QMutexLocker locker(&mutex);
    while (!redoStack.empty()) {
        redoStack.pop();
    }
}

const cv::Mat& ImageProcessor::getLastProcessedImage() const
{
    return lastProcessedImage;
}

void ImageProcessor::pushToUndoStack(const cv::Mat& image)
{
    undoStack.push(image.clone());
}

void ImageProcessor::pushToRedoStack(const cv::Mat& image)
{
    redoStack.push(image.clone());
}

cv::Mat ImageProcessor::convertToGrayScale(const cv::Mat& image)
{
    // RGB ä���� �׷��̽����Ϸ� ��ȯ
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // �׷��̽����� �̹����� RGB �������� ��ȯ
    cv::Mat merged;
    cv::merge(std::vector<cv::Mat>{gray, gray, gray}, merged);

    return merged;
}
