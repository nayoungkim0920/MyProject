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

            //이미지 회전
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
                qDebug() << "입력 이미지가 비어 있습니다.";
                return false;
            }

            if (scaleFactor <= 0) {
                qDebug() << "잘못된 확대/축소 배율입니다.";
                return false;
            }            

            int newWidth = static_cast<int>(image.cols * scaleFactor);
            int newHeight = static_cast<int>(image.rows * scaleFactor);

            cv::Mat zoomedImage;
            cv::resize(image, zoomedImage, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
            
            image = zoomedImage.clone(); // 이미지를 복사하여 업데이트
            lastProcessedImage = image.clone();

            emit imageProcessed(image); // 이미지 처리 완료 시그널 발생

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "이미지 확대/축소 중 예외가 발생했습니다:" << e.what();
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

            // 원본 이미지에 그레이스케일 이미지 적용
            image = gray.clone();
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

            //회색조이미지
            cv::Mat gray = convertToGrayScale(image);

            cv::Mat edges;//엣지감지결과
            //50:하위 임계값(threshold1)
            //엣지후보로 고려되기 위한 최소 강도 
            //임계값보다 낮은 강도의 엣지는 제거된다.
            //150:상위 임계값(threshold2)
            //강한 엣지로 간주되는 임계값
            //두 임계값 사이의 강도를 가진 엣지는 연결되어
            //하나의 엣지로 유지된다
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
                qDebug() << "median 필터를 적용할 이미지가 없습니다.";
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
            qDebug() << "median 필터 적용 중 오류 발생: "
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
                qDebug() << "laplacian 필터를 적용할 이미지가 없습니다.";
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
            qDebug() << "laplacian 필터 적용 중 오류 발생: "
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
                qDebug() << "bilateral 필터를 적용할 이미지가 없습니다.";
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
            qDebug() << "bilateral 필터 적용 중 오류 발생: "
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

//실행취소
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

//재실행
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
    // RGB 채널을 그레이스케일로 변환
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // 그레이스케일 이미지를 RGB 형식으로 변환
    cv::Mat merged;
    cv::merge(std::vector<cv::Mat>{gray, gray, gray}, merged);

    return merged;
}
