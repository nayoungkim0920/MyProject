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


// QDebug에서 cv::Size를 출력할 수 있도록 변환 함수 작성
QDebug operator<<(QDebug dbg, const cv::Size& size) {
    dbg.nospace() << "Size(width=" << size.width << ", height=" << size.height << ")";
    return dbg.space();
}

// QDebug에서 cv::Mat의 타입을 출력할 수 있도록 변환 함수 작성
QDebug operator<<(QDebug dbg, const cv::Mat& mat) {
    dbg.nospace() << "Mat(type=" << mat.type() << ", size=" << mat.size() << ")";
    return dbg.space();
}

QFuture<bool> ImageProcessor::convertToGrayscaleAsync(cv::Mat& image)
{
    return QtConcurrent::run([this, &image]() -> bool {

        QMutexLocker locker(&mutex);

        try {
            if (image.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }
            
            if (image.channels() == 1) {
                qDebug() << "Input image is already a grayscale image.";
                return true; // 이미 그레이스케일이므로 처리하지 않음
            }

            if (image.channels() != 3) {
                qDebug() << "Input image is not a 3-channel BGR image.";
                return false;
            }

            qDebug() << "Original image size:" << image.size() << "type:" << image.type();

            pushToUndoStack(image);

            // CUDA 장치 설정
            cv::cuda::setDevice(0);
            qDebug() << "CUDA device set.";

            // 입력 이미지를 CUDA GpuMat으로 업로드
            cv::cuda::GpuMat d_input;
            d_input.upload(image);
            qDebug() << "Image uploaded to CUDA. Size:" << d_input.size() << "type:" << d_input.type();

            // CUDA를 사용하여 그레이스케일로 변환
            cv::cuda::GpuMat d_output;
            cv::cuda::cvtColor(d_input, d_output, cv::COLOR_BGR2GRAY);
            qDebug() << "Image converted to grayscale on CUDA. Size:" << d_output.size() << "type:" << d_output.type();

            // CUDA에서 호스트로 이미지 다운로드
            cv::Mat output;
            d_output.download(output);
            qDebug() << "Image downloaded from CUDA. Size:" << output.size() << "type:" << output.type();

            if (output.empty()) {
                qDebug() << "Output image is empty after CUDA processing.";
                return false;
            }

            if (output.type() != CV_8UC1) {
                qDebug() << "Output image type is not CV_8UC1, something went wrong.";
                return false;
            }

            // 원본 이미지에 그레이스케일 이미지 적용
            image = output.clone();
            lastProcessedImage = image.clone();

            emit imageProcessed(image);
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
            if (image.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            pushToUndoStack(image);

            cv::cuda::GpuMat d_input;
            if (image.channels() == 1) {
                // 이미 그레이스케일인 경우
                d_input.upload(image);
            }
            else {
                // BGR 이미지인 경우 그레이스케일로 변환 후 업로드
                cv::Mat grayImage;
                cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
                d_input.upload(grayImage);
            }

            // 캐니 엣지 감지기 생성 및 적용
            cv::Ptr<cv::cuda::CannyEdgeDetector> cannyDetector = cv::cuda::createCannyEdgeDetector(50, 150);
            cv::cuda::GpuMat d_cannyEdges;
            cannyDetector->detect(d_input, d_cannyEdges);

            // 결과를 CPU 메모리로 복사
            cv::Mat edges;
            d_cannyEdges.download(edges);

            // 출력 이미지에 엣지 표시 (예시)
            cv::Mat outputImage = image.clone();
            outputImage.setTo(cv::Scalar(0, 255, 0), edges); // Green edges

            image = outputImage.clone();
            lastProcessedImage = image.clone();

            emit imageProcessed(image);

            // GPU 메모리 해제
            d_input.release();
            d_cannyEdges.release();

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