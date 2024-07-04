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
    // �Լ� �̸��� ���ڿ��� ����
    const char* functionName = __func__;

    return QtConcurrent::run([this, &image, functionName]() -> bool {

        QMutexLocker locker(&mutex);

        try {

            if (image.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            pushToUndoStack(image);

            // ó���ð� ��� (�ӽ÷� �ð� ������ �־����ϴ�)
            double startTime = getCurrentTimeMs();

            // �̹����� CUDA�� �̿��Ͽ� ȸ��
            // imageProcessing.cuh/imagProessing.cu            
            //callRotateImageCUDA(image);

            // CUDA �����Լ���  ����
            // #include <opencv2/cudawarping.hpp>
            double angle = 90.0; // ȸ���� ���� (��: 90��)

            // �̹����� GPU �޸𸮿� ���ε�
            cv::cuda::GpuMat gpuImage;
            gpuImage.upload(image);

            // ȸ�� �߽��� �̹����� �߾����� ����
            cv::Point2f center(gpuImage.cols / 2.0f, gpuImage.rows / 2.0f);

            // ȸ�� ��Ʈ���� ���
            cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);

            // GPU���� ȸ�� ����
            cv::cuda::GpuMat gpuRotatedImage;
            cv::cuda::warpAffine(gpuImage, gpuRotatedImage, rotationMatrix, gpuImage.size());

            // ��� �̹����� CPU �޸𸮷� �ٿ�ε�
            gpuRotatedImage.download(image);

            // �̹��� ó�� ��

            // �̹��� ó�� �ð� ����
            double endTime = getCurrentTimeMs();
            double processingTime = endTime - startTime;

            // �̹��� ������Ʈ �� �ñ׳� �߻�
            emit imageProcessed(image, processingTime, functionName);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "Exception occurred while rotating image:" << e.what();
            return false;
        }
        });
}

QFuture<bool> ImageProcessor::zoomoutImage(cv::Mat& image, double scaleFactor)
{
    //�Լ� �̸��� ���ڿ��� ����
    const char* functionName = __func__;

    return QtConcurrent::run([this, &image, scaleFactor, functionName]() -> bool {

        QMutexLocker locker(&mutex);

        try {            

            if (image.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            if (scaleFactor <= 0) {
                qDebug() << "�߸��� ��� �����Դϴ�.";
                return false;
            }

            pushToUndoStack(image);

            // ó���ð���� ����
            double startTime = getCurrentTimeMs();

            int newWidth = static_cast<int>(image.cols * scaleFactor);
            int newHeight = static_cast<int>(image.rows * scaleFactor);

            cv::Mat zoomedImage;
            cv::resize(image, zoomedImage, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);

            // ó���ð���� ����
            double endTime = getCurrentTimeMs();
            double processingTime = endTime - startTime;

            image = zoomedImage.clone(); // �̹����� �����Ͽ� ������Ʈ
            lastProcessedImage = image.clone();

            emit imageProcessed(image, processingTime, functionName); // �̹��� ó�� �Ϸ� �ñ׳� �߻�

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "�̹��� ��� �� ���ܰ� �߻��߽��ϴ�:" << e.what();
            return false;
        }
        });
}

QFuture<bool> ImageProcessor::zoominImage(cv::Mat& image, double scaleFactor)
{
    //�Լ� �̸��� ���ڿ��� ����
    const char* functionName = __func__;

    return QtConcurrent::run([this, &image, scaleFactor, functionName]() -> bool {

        QMutexLocker locker(&mutex);

        try {

            if (image.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            if (scaleFactor <= 0) {
                qDebug() << "�߸��� Ȯ�� �����Դϴ�.";
                return false;
            }

            pushToUndoStack(image);

            // ó���ð���� ����
            double startTime = getCurrentTimeMs();

            int newWidth = static_cast<int>(image.cols * scaleFactor);
            int newHeight = static_cast<int>(image.rows * scaleFactor);

            cv::Mat zoomedImage;
            cv::resize(image, zoomedImage, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);

            // ó���ð���� ����
            double endTime = getCurrentTimeMs();
            double processingTime = endTime - startTime;

            image = zoomedImage.clone(); // �̹����� �����Ͽ� ������Ʈ
            lastProcessedImage = image.clone();

            emit imageProcessed(image, processingTime, functionName); // �̹��� ó�� �Ϸ� �ñ׳� �߻�

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "�̹��� Ȯ�� �� ���ܰ� �߻��߽��ϴ�:" << e.what();
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

QFuture<bool> ImageProcessor::grayScale(cv::Mat& image)
{   
    //�Լ� �̸��� ���ڿ��� ����
    const char* functionName = __func__;

    return QtConcurrent::run([this, &image, functionName]() -> bool {
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

            // ó���ð���� ����
            double startTime = getCurrentTimeMs();

            if (!grayScaleCUDA(image)) {
                return false;
            }

            // ó���ð���� ����
            double endTime = getCurrentTimeMs();
            double processingTime = endTime - startTime;

            emit imageProcessed(image, processingTime, functionName); // ��ȯ�� �̹��� ��ȣ ����

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "Exception occurred while converting to grayscale:" << e.what();
            return false;
        }
        });
}

bool ImageProcessor::grayScaleCUDA(cv::Mat& image)
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

double ImageProcessor::getCurrentTimeMs()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::steady_clock::now().time_since_epoch()).count();
}

QFuture<bool> ImageProcessor::gaussianBlur(cv::Mat& image, int kernelSize)
{
    //�Լ� �̸��� ���ڿ��� ����
    const char* functionName = __func__;

    return QtConcurrent::run([this, &image, kernelSize, functionName]() -> bool {

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

            // ó���ð���� ����
            double startTime = getCurrentTimeMs();

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

            // ó���ð���� ����
            double endTime = getCurrentTimeMs();
            double processingTime = endTime - startTime;            

            image = blurredImage.clone();
            lastProcessedImage = image.clone();

            emit imageProcessed(image, processingTime, functionName);

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
    //�Լ� �̸��� ���ڿ��� ����
    const char* functionName = __func__;

    return QtConcurrent::run([this, &image, functionName]() -> bool {
        QMutexLocker locker(&mutex);

        try {

            if (image.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            pushToUndoStack(image);

            // ó���ð���� ����
            double startTime = getCurrentTimeMs();

            //�׷��̽������� �ƴѰ��
            if (image.channels() != 1)
            {
                if (!grayScaleCUDA(image)) {
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

            // ó���ð���� ����
            double endTime = getCurrentTimeMs();
            double processingTime = endTime - startTime;

            image = outputImage.clone();
            lastProcessedImage = image.clone();

            // GPU �޸� ����
            d_cannyEdges.release();

            emit imageProcessed(image, processingTime, functionName);

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

    //�Լ� �̸��� ���ڿ��� ����
    const char* functionName = __func__;

    return QtConcurrent::run([this, &image, functionName]()->bool {

        QMutexLocker locker(&mutex);

        try {

            if (image.empty()) {
                qDebug() << "median ���͸� ������ �̹����� �����ϴ�.";
                return false;
            }

            pushToUndoStack(image);

            // ó���ð���� ����
            double startTime = getCurrentTimeMs();

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

            // ó���ð���� ����
            double endTime = getCurrentTimeMs();
            double processingTime = endTime - startTime;

            image = medianedImage.clone();
            lastProcessedImage = image.clone();

            emit imageProcessed(image, processingTime, functionName);

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
    //�Լ� �̸��� ���ڿ��� ����
    const char* functionName = __func__;

    return QtConcurrent::run([this, &image, functionName]()->bool {

        QMutexLocker locker(&mutex);

        try {            

            if (image.empty()) {
                qDebug() << "laplacian ���͸� ������ �̹����� �����ϴ�.";
                return false;
            }

            pushToUndoStack(image);

            // ó���ð���� ����
            double startTime = getCurrentTimeMs();

            cv::Mat filteredImage;
            cv::Laplacian(image, filteredImage, CV_8U, 3);

            // ó���ð���� ����
            double endTime = getCurrentTimeMs();
            double processingTime = endTime - startTime;

            image = filteredImage.clone();
            lastProcessedImage = image.clone();

            emit imageProcessed(image, processingTime, functionName);

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
    //�Լ� �̸��� ���ڿ��� ����
    const char* functionName = __func__;

    return QtConcurrent::run([this, &image, functionName]()->bool {

        QMutexLocker locker(&mutex);

        try {               

            if (image.empty()) {
                qDebug() << "bilateral ���͸� ������ �̹����� �����ϴ�.";
                return false;
            }

            pushToUndoStack(image);

            // ó���ð���� ����
            double startTime = getCurrentTimeMs();

            cv::Mat filteredImage;
            cv::bilateralFilter(image, filteredImage, 9, 75, 75);

            // ó���ð���� ����
            double endTime = getCurrentTimeMs();
            double processingTime = endTime - startTime;

            image = filteredImage.clone();
            lastProcessedImage = image.clone();

            emit imageProcessed(image, processingTime, functionName);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "bilateral ���� ���� �� ���� �߻�: "
                << e.what();
            return false;
        }
        });
}

QFuture<bool> ImageProcessor::sobelFilter(cv::Mat& image)
{
    //�Լ� �̸��� ���ڿ��� ����
    const char* functionName = __func__;

    return QtConcurrent::run([this, &image, functionName]()->bool {
        if (cv::cuda::getCudaEnabledDeviceCount() <= 0) {
            qDebug() << "No CUDA-enabled device found. Falling back to CPU implementation.";
            return false;
        }

        pushToUndoStack(image);

        // ó���ð���� ����
        double startTime = getCurrentTimeMs();

        cv::cuda::GpuMat gpuImage, gpuGray, gpuSobelX, gpuSobelY;
        gpuImage.upload(image);
        cv::cuda::cvtColor(gpuImage, gpuGray, cv::COLOR_BGR2GRAY);

        cv::Ptr<cv::cuda::Filter> sobelX =
            cv::cuda::createSobelFilter(gpuGray.type(), CV_16S, 1, 0);
        cv::Ptr<cv::cuda::Filter> sobelY =
            cv::cuda::createSobelFilter(gpuGray.type(), CV_16S, 0, 1);

        sobelX->apply(gpuGray, gpuSobelX);
        sobelY->apply(gpuGray, gpuSobelY);

        cv::cuda::GpuMat sobelX_8U, sobelY_8U;
        gpuSobelX.convertTo(sobelX_8U, CV_8U);
        gpuSobelY.convertTo(sobelY_8U, CV_8U);

        cv::cuda::addWeighted(sobelX_8U, 0.5, sobelY_8U, 0.5 ,0, gpuGray);

        cv::Mat sobeledImage;
        gpuGray.download(sobeledImage);

        // ó���ð���� ����
        double endTime = getCurrentTimeMs();
        double processingTime = endTime - startTime;

        image = sobeledImage.clone();
        lastProcessedImage = image.clone();

        emit imageProcessed(image, processingTime, functionName);

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
    //�Լ� �̸��� ���ڿ��� ����
    const char* functionName = __func__;

    try {
        if (!canUndo()) {
            throw std::runtime_error("Cannot undo: Undo stack is empty");
        }

        // ó���ð���� ����
        double startTime = getCurrentTimeMs();

        // Push the current image to the redo stack
        redoStack.push(lastProcessedImage);

        // Retrieve the image to restore from the undo stack and assign it to lastProcessedImage
        cv::Mat imageToRestore = undoStack.top();
        lastProcessedImage = imageToRestore.clone();

        // Remove the image from the undo stack
        undoStack.pop();

        // ó���ð���� ����
        double endTime = getCurrentTimeMs();
        double processingTime = endTime - startTime;

        // Emit signal indicating image processing is complete
        emit imageProcessed(lastProcessedImage, processingTime, functionName);
    }
    catch (const std::exception& e) {
        qDebug() << "Exception occurred in ImageProcessor::undo(): " << e.what();
    }
}




//�����
void ImageProcessor::redo()
{
    //�Լ� �̸��� ���ڿ��� ����
    const char* functionName = __func__;

    try {

        if (!canRedo())
            throw std::runtime_error("Cannot redo: redo stack is empty");

        // ó���ð���� ����
        double startTime = getCurrentTimeMs();

        cv::Mat imageToRestore = redoStack.top();
        undoStack.push(lastProcessedImage);
        lastProcessedImage = imageToRestore.clone();
        redoStack.pop();

        // ó���ð���� ����
        double endTime = getCurrentTimeMs();
        double processingTime = endTime - startTime;

        emit imageProcessed(lastProcessedImage, processingTime, functionName);

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

void ImageProcessor::initializeCUDA()
{
    // ������ ���� �۾��� �����Ͽ� CUDA �ʱ�ȭ�� ����
    cv::cuda::GpuMat temp;
    temp.upload(cv::Mat::zeros(1, 1, CV_8UC1));
    cv::cuda::cvtColor(temp, temp, cv::COLOR_GRAY2BGR);
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
