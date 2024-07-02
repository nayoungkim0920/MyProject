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


// QDebug���� cv::Size�� ����� �� �ֵ��� ��ȯ �Լ� �ۼ�
QDebug operator<<(QDebug dbg, const cv::Size& size) {
    dbg.nospace() << "Size(width=" << size.width << ", height=" << size.height << ")";
    return dbg.space();
}

// QDebug���� cv::Mat�� Ÿ���� ����� �� �ֵ��� ��ȯ �Լ� �ۼ�
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

            if (image.channels() != 3) {
                pushToUndoStack(image);
                qDebug() << "Input image is not a 3-channel BGR image.";
                return false;
            }

            if (image.channels() == 1) {
                pushToUndoStack(image);
                qDebug() << "Input image is already a grayscale image.";
                return false; // �̹� �׷��̽������̹Ƿ� ó������ ����
            }

            pushToUndoStack(image);

            if (!convertToGrayscaleCUDA(image)) {
                return false;
            }

            emit imageProcessed(image); // ��ȯ�� �̹��� ��ȣ ����

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "Exception occurred while converting to grayscale:" << e.what();
            return false;
        }
        });
}

bool ImageProcessor::convertToGrayscaleCUDA(cv::Mat& image)
{
    try {

        // CUDA ��ġ ����
        cv::cuda::setDevice(0);

        // �Է� �̹����� CUDA GpuMat���� ���ε�
        cv::cuda::GpuMat d_input;
        d_input.upload(image);

        // CUDA�� ����Ͽ� �׷��̽����Ϸ� ��ȯ
        cv::cuda::GpuMat d_output;
        cv::cuda::cvtColor(d_input, d_output, cv::COLOR_BGR2GRAY);

        // CUDA���� ȣ��Ʈ�� �̹��� �ٿ�ε�
        cv::Mat output;
        d_output.download(output);

        if (output.empty() || output.type() != CV_8UC1) {
            qDebug() << "Output image is empty or not in expected format after CUDA processing.";
            return false;
        }

        // ���� �̹����� �׷��̽����� �̹����� ������Ʈ
        image = output.clone(); // ��ȯ�� �׷��̽����� �̹����� ������Ʈ
        lastProcessedImage = image.clone(); // ������ ó���� �̹��� ������Ʈ

        return true;
    }
    catch (const cv::Exception& e) {
        qDebug() << "Exception occurred while converting to grayscale using CUDA:" << e.what();
        return false;
    }
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

            pushToUndoStack(image);

            // Upload image to GPU
            cv::cuda::GpuMat gpuImage;
            gpuImage.upload(image);

            // Create Gaussian filter
            cv::Ptr<cv::cuda::Filter> gaussianFilter =
                cv::cuda::createGaussianFilter(
                    gpuImage.type(),
                    gpuImage.type(),
                    cv::Size(kernelSize, kernelSize),
                    0);

            // Apply Gaussian blur on GPU
            cv::cuda::GpuMat blurredGpuImage;
            gaussianFilter->apply(gpuImage, blurredGpuImage);

            // Download the result back to CPU
            cv::Mat blurredImage;
            blurredGpuImage.download(blurredImage);

            image = blurredImage.clone();
            lastProcessedImage = image.clone();

            emit imageProcessed(image);

            return true;

            /* OpenCV
            cv::Mat blurredImage;
            cv::GaussianBlur(image, blurredImage, cv::Size(kernelSize, kernelSize), 0, 0);

            image = blurredImage.clone();
            lastProcessedImage = image.clone();

            emit imageProcessed(image);

            return true;
            */
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

            //�׷��̽������� �ƴѰ��
            if (image.channels() != 1)
            {
                if (!convertToGrayscaleCUDA(image)) {
                    return false;
                }
            }

            // GPU���� ĳ�� ���� ������ ����
            cv::cuda::GpuMat d_input(image);
            cv::cuda::GpuMat d_cannyEdges;
            cv::Ptr<cv::cuda::CannyEdgeDetector> cannyDetector = cv::cuda::createCannyEdgeDetector(50, 150);
            cannyDetector->detect(d_input, d_cannyEdges);

            // ����� CPU �޸𸮷� ����
            cv::Mat edges;
            d_cannyEdges.download(edges);

            // ��� �̹����� �ʷϻ� ���� ǥ��
            cv::Mat outputImage = cv::Mat::zeros(image.size(), CV_8UC3); // 3-channel BGR image
            cv::Mat mask(edges.size(), CV_8UC1, cv::Scalar(0)); // Mask for green edges
            mask.setTo(cv::Scalar(255), edges); // Set pixels to 255 (white) where edges are detected
            cv::Mat channels[3];
            cv::split(outputImage, channels);
            channels[1] = mask; // Green channel is set by mask
            cv::merge(channels, 3, outputImage); // Merge channels to get green edges

            image = outputImage.clone();
            lastProcessedImage = image.clone();

            // GPU �޸� ����
            d_cannyEdges.release();

            emit imageProcessed(image);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "Exception occurred while applying Canny edges:" << e.what();
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

            // Upload image to GPU
            cv::cuda::GpuMat gpuImage;
            gpuImage.upload(image);

            // Create median filter
            cv::Ptr<cv::cuda::Filter> medianFilter =
                cv::cuda::createMedianFilter(gpuImage.type(), 5);

            // Apply median filter on GPU
            cv::cuda::GpuMat medianedGpuImage;
            medianFilter->apply(gpuImage, medianedGpuImage);

            // Download the result back to CPU
            cv::Mat medianedImage;
            medianedGpuImage.download(medianedImage);

            image = medianedImage.clone();
            lastProcessedImage = image.clone();

            emit imageProcessed(image);

            return true;

            /*
            cv::Mat medianedImage;
            cv::medianBlur(image, medianedImage, 5);
            image = medianedImage.clone();
            lastProcessedImage = image.clone();

            emit imageProcessed(image);

            return true;
            */
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
// Undo operation
void ImageProcessor::undo()
{
    try {
        if (!canUndo()) {
            throw std::runtime_error("Cannot undo: Undo stack is empty");
        }

        // Push the current image to the redo stack
        redoStack.push(lastProcessedImage);

        // Retrieve the image to restore from the undo stack and assign it to lastProcessedImage
        cv::Mat imageToRestore = undoStack.top();
        lastProcessedImage = imageToRestore.clone();

        // Remove the image from the undo stack
        undoStack.pop();

        // Emit signal indicating image processing is complete
        emit imageProcessed(lastProcessedImage);
    }
    catch (const std::exception& e) {
        qDebug() << "Exception occurred in ImageProcessor::undo(): " << e.what();
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
