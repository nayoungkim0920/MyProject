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

QFuture<bool> ImageProcessor::rotateImage(cv::Mat& imageOpenCV
                                        , cv::Mat& imageIPP
                                        , cv::Mat& imageCUDA
                                        , cv::Mat& imageCUDAKernel)
{
    //�Լ� �̸��� ���ڿ��� ����
    const char* functionName = __func__;

    return QtConcurrent::run([this
        , &imageOpenCV
        , &imageIPP
        , &imageCUDA
        , &imageCUDAKernel
        , functionName]() -> bool {

        QMutexLocker locker(&mutex);

        try {

            if (imageOpenCV.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            pushToUndoStackOpenCV(imageOpenCV.clone());
            pushToUndoStackIPP(imageIPP.clone());
            pushToUndoStackCUDA(imageCUDA.clone());
            pushToUndoStackCUDAKernel(imageCUDAKernel.clone());

            QVector<ProcessingResult> results;

            ProcessingResult outputOpenCV = rotateOpenCV(imageOpenCV);
            lastProcessedImageOpenCV = outputOpenCV.processedImage.clone();
            results.append(outputOpenCV);

            ProcessingResult outputIPP = rotateIPP(imageIPP);
            lastProcessedImageIPP = outputIPP.processedImage.clone();
            results.append(outputIPP);

            ProcessingResult outputCUDA = rotateCUDA(imageCUDA);
            lastProcessedImageCUDA = outputCUDA.processedImage.clone();
            results.append(outputCUDA);

            ProcessingResult outputCUDAKernel = rotateCUDAKernel(imageCUDAKernel);
            lastProcessedImageCUDAKernel = outputCUDAKernel.processedImage.clone();
            results.append(outputCUDAKernel);

            // �̹��� ������Ʈ �� �ñ׳� �߻�
            emit imageProcessed(results);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "Exception occurred while rotating image:" << e.what();
            return false;
        }
        });
}

QFuture<bool> ImageProcessor::zoomOutImage(cv::Mat& imageOpenCV
                                            , cv::Mat& imageIPP
                                            , cv::Mat& imageCUDA
                                            , cv::Mat& imageCUDAKernel
                                            , double scaleFactor)
{
    //�Լ� �̸��� ���ڿ��� ����
    const char* functionName = __func__;

    return QtConcurrent::run([this
                            , &imageOpenCV
                            , &imageIPP
                            , &imageCUDA
                            , &imageCUDAKernel
                            , scaleFactor
                            , functionName]() -> bool {

            QMutexLocker locker(&mutex);

            try {

                if (imageOpenCV.empty()) {
                    qDebug() << "Input image is empty.";
                    return false;
                }

                if (scaleFactor <= 0) {
                    qDebug() << "Invalid scaling factor.";
                    return false;
                }

                pushToUndoStackOpenCV(imageOpenCV.clone());
                pushToUndoStackIPP(imageIPP.clone());
                pushToUndoStackCUDA(imageCUDA.clone());
                pushToUndoStackCUDAKernel(imageCUDAKernel.clone());

                QVector<ProcessingResult> results;

                double newWidth = static_cast<int>(imageOpenCV.cols * scaleFactor);
                double newHeight = static_cast<int>(imageOpenCV.rows * scaleFactor);

                ProcessingResult outputOpenCV = zoomOpenCV(imageOpenCV, newWidth, newHeight);
                lastProcessedImageOpenCV = outputOpenCV.processedImage.clone();
                results.append(outputOpenCV);

                ProcessingResult outputIPP = zoomIPP(imageIPP, newWidth, newHeight);
                lastProcessedImageIPP = outputIPP.processedImage.clone();
                results.append(outputIPP);

                ProcessingResult outputCUDA = zoomCUDA(imageCUDA, newWidth, newHeight);
                lastProcessedImageCUDA = outputCUDA.processedImage.clone();
                results.append(outputCUDA);

                ProcessingResult outputCUDAKernel = zoomCUDAKernel(imageCUDAKernel, newWidth, newHeight);
                lastProcessedImageCUDAKernel = outputCUDAKernel.processedImage.clone();
                results.append(outputCUDAKernel);

                emit imageProcessed(results); // �̹��� ó�� �Ϸ� �ñ׳� �߻�

                return true;
            }
            catch (const cv::Exception& e) {
                qDebug() << "�̹��� ��� �� ���ܰ� �߻��߽��ϴ�:" << e.what();
                return false;
            }
        });
}

QFuture<bool> ImageProcessor::zoomInImage(cv::Mat& imageOpenCV
                                        , cv::Mat& imageIPP
                                        , cv::Mat& imageCUDA
                                        , cv::Mat& imageCUDAKernel
                                        , double scaleFactor)
{
    //�Լ� �̸��� ���ڿ��� ����
    const char* functionName = __func__;

    return QtConcurrent::run([this
                            , &imageOpenCV
                            , &imageIPP
                            , &imageCUDA
                            , &imageCUDAKernel
                            , scaleFactor
                            , functionName]() -> bool {

        QMutexLocker locker(&mutex);

        try {

            if (imageOpenCV.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            if (scaleFactor <= 0) {
                qDebug() << "�߸��� Ȯ�� �����Դϴ�.";
                return false;
            }

            pushToUndoStackOpenCV(imageOpenCV.clone());
            pushToUndoStackIPP(imageIPP.clone());
            pushToUndoStackCUDA(imageCUDA.clone());
            pushToUndoStackCUDAKernel(imageCUDAKernel.clone());

            QVector<ProcessingResult> results;

            double newWidth = static_cast<int>(imageOpenCV.cols * scaleFactor);
            double newHeight = static_cast<int>(imageOpenCV.rows * scaleFactor);

            ProcessingResult outputOpenCV = zoomOpenCV(imageOpenCV, newWidth, newHeight);
            lastProcessedImageOpenCV = outputOpenCV.processedImage.clone();
            results.append(outputOpenCV);

            ProcessingResult outputIPP = zoomIPP(imageIPP, newWidth, newHeight);
            lastProcessedImageIPP = outputIPP.processedImage.clone();
            results.append(outputIPP);

            ProcessingResult outputCUDA = zoomCUDA(imageCUDA, newWidth, newHeight);
            lastProcessedImageCUDA = outputCUDA.processedImage.clone();
            results.append(outputCUDA);

            ProcessingResult outputCUDAKernel = zoomCUDAKernel(imageCUDAKernel, newWidth, newHeight);
            lastProcessedImageCUDAKernel = outputCUDAKernel.processedImage.clone();
            results.append(outputCUDAKernel);

            emit imageProcessed(results); // �̹��� ó�� �Ϸ� �ñ׳� �߻�

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

QFuture<bool> ImageProcessor::grayScale(cv::Mat& imageOpenCV
                                        , cv::Mat& imageIPP
                                        , cv::Mat& imageCUDA
                                        , cv::Mat& imageCUDAKernel)
{
    const char* functionName = __func__;

    return QtConcurrent::run([this
                            , &imageOpenCV
                            , &imageIPP
                            , &imageCUDA
                            , &imageCUDAKernel
                            , functionName]() -> bool {
        
        QMutexLocker locker(&mutex);

        try {
            if (imageOpenCV.channels() != 3 && imageOpenCV.channels() != 1) {
                qDebug() << "Input image must be a 3-channel BGR image or already grayscale.";
                return false;
            }

            if (imageOpenCV.channels() == 3) {
                pushToUndoStackOpenCV(imageOpenCV.clone());
                pushToUndoStackIPP(imageIPP.clone());
                pushToUndoStackCUDA(imageCUDA.clone());
                pushToUndoStackCUDAKernel(imageCUDAKernel.clone());

                QVector<ProcessingResult> results;

                ProcessingResult outputOpenCV = grayScaleOpenCV(imageOpenCV);
                lastProcessedImageOpenCV = outputOpenCV.processedImage.clone();
                results.append(outputOpenCV);

                ProcessingResult outputIPP = grayScaleIPP(imageIPP);
                lastProcessedImageIPP = outputIPP.processedImage.clone();
                results.append(outputIPP);

                ProcessingResult outputCUDA = grayScaleCUDA(imageCUDA);
                lastProcessedImageCUDA = outputCUDA.processedImage.clone();
                results.append(outputCUDA);

                ProcessingResult outputCUDAKernel = grayScaleCUDAKernel(imageCUDAKernel);
                lastProcessedImageCUDAKernel = outputCUDAKernel.processedImage.clone();
                results.append(outputCUDAKernel);

                emit imageProcessed(results);
            }
            else {
                pushToUndoStackOpenCV(imageOpenCV.clone());
                pushToUndoStackIPP(imageIPP.clone());
                pushToUndoStackCUDA(imageCUDA.clone());
                pushToUndoStackCUDAKernel(imageCUDAKernel.clone());

                lastProcessedImageOpenCV = imageOpenCV.clone();
                lastProcessedImageIPP = imageIPP.clone();
                lastProcessedImageCUDA = imageCUDA.clone();
                lastProcessedImageCUDAKernel = imageCUDAKernel.clone();
            }

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "Exception occurred while converting to grayscale:" << e.what();
            return false;
        }
        });
}

/*bool ImageProcessor::grayScaleCUDA(cv::Mat& image)
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
}*/

ImageProcessor::ProcessingResult ImageProcessor::grayScaleOpenCV(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    ImageProcessorOpenCV IPOpenCV;
    cv::Mat outputImage = IPOpenCV.grayScale(inputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "grayScale", "OpenCV", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::grayScaleIPP(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    ImageProcessorIPP IPIPP;
    cv::Mat outputImage = IPIPP.grayScale(inputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���   

    result = setResult(result, inputImage, outputImage, "grayScale", "IPP", elapsedTimeMs);

    return result;
}

cv::Mat ImageProcessor::convertToGrayOpenCV(const cv::Mat& inputImage)
{
    cv::Mat grayImage;
    cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);

    return grayImage;
}

cv::Mat ImageProcessor::convertToGrayIPP(const cv::Mat& inputImage)
{
    // IPP �ʱ�ȭ
    ippInit();

    // �Է� �̹����� ũ�� �� ���� ����
    IppiSize roiSize = { inputImage.cols, inputImage.rows };
    int srcStep = inputImage.step;
    int dstStep = inputImage.cols;
    Ipp8u* srcData = inputImage.data;

    // ��� �̹��� ���� �� IPP �޸� �Ҵ�
    cv::Mat grayImage(inputImage.rows, inputImage.cols, CV_8UC1);
    Ipp8u* dstData = grayImage.data;

    // IPP RGB to Gray ��ȯ ����
    IppStatus status = ippiRGBToGray_8u_C3C1R(srcData, srcStep, dstData, dstStep, roiSize);
    if (status != ippStsNoErr) {
        std::cerr << "IPP ����: " << status << std::endl;
        return cv::Mat(); // ���� �߻� �� �� Mat ��ȯ
    }

    return grayImage;
}

cv::cuda::GpuMat ImageProcessor::convertToGrayCUDA(const cv::cuda::GpuMat& inputImage)
{
    cv::cuda::GpuMat grayImage;
    cv::cuda::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);

    return grayImage;
}

cv::Mat ImageProcessor::convertToGrayCUDAKernel(cv::Mat& inputImage)
{
    cv::Mat grayImage;
    callGrayScaleImageCUDA(inputImage, grayImage);

    return grayImage;
}

ImageProcessor::ProcessingResult ImageProcessor::grayScaleCUDA(cv::Mat& inputImage)
{    
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    cv::cuda::GpuMat d_inputImage;
    d_inputImage.upload(inputImage);

    cv::Mat outputImage;
    cv::cuda::GpuMat d_outputImage = convertToGrayCUDA(d_inputImage);
    d_outputImage.download(outputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "grayScale", "CUDA", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::grayScaleCUDAKernel(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    cv::Mat outputImage = convertToGrayCUDAKernel(inputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "grayScale", "CUDAKernel", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::zoomOpenCV(cv::Mat& inputImage, double newWidth, double newHeight)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    ImageProcessorOpenCV IPOpenCV;
    cv::Mat outputImage;
    outputImage = IPOpenCV.zoom(inputImage, newWidth, newHeight, 0, 0, cv::INTER_LINEAR);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "zoom", "OpenCV", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::zoomIPP(cv::Mat& inputImage, double newWidth, double newHeight) {
 
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    ImageProcessorIPP IPIPP;
    cv::Mat outputImage = IPIPP.zoom(inputImage, newWidth, newHeight, 0, 0, 1);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "zoom", "IPP", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::zoomCUDA(cv::Mat& inputImage, double newWidth, double newHeight)
{    
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    // GPU �޸𸮷� �̹��� ���ε�
    cv::cuda::GpuMat d_image;
    d_image.upload(inputImage);

    // ��� �̹����� ������ GPU �޸� �Ҵ�
    cv::cuda::GpuMat d_zoomInImage;

    // �̹��� ũ�� ����
    cv::cuda::resize(d_image, d_zoomInImage, cv::Size(static_cast<int>(newWidth), static_cast<int>(newHeight)), 0, 0, cv::INTER_LINEAR);

    // CPU �޸𸮷� ��� �̹��� �ٿ�ε�
    cv::Mat outputImage;
    d_zoomInImage.download(outputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "zoom", "CUDA", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::zoomCUDAKernel(cv::Mat& inputImage, double newWidth, double newHeight)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    cv::Mat outputImage;
    callZoomImageCUDA(inputImage, outputImage, newWidth, newHeight);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "zoom", "CUDAKernel", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::rotateOpenCV(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    ImageProcessorOpenCV IPOpenCV;
    cv::Mat outputImage = IPOpenCV.rotate(inputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "rotate", "OpenCV", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::rotateIPP(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount();

    ImageProcessorIPP IPIPP;
    cv::Mat outputImage = IPIPP.rotate(inputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "rotate", "IPP", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::rotateCUDA(cv::Mat& inputImage)
{
    ProcessingResult result;

    double startTime = cv::getTickCount(); // ���� �ð� ����

    // #include <opencv2/cudawarping.hpp>
    double angle = 90.0; // ȸ���� ���� (��: 90��)

    // �̹����� GPU �޸𸮿� ���ε�
    cv::cuda::GpuMat gpuImage;
    gpuImage.upload(inputImage);

    // ȸ�� �߽��� �̹����� �߾����� ����
    cv::Point2f center(gpuImage.cols / 2.0f, gpuImage.rows / 2.0f);

    // ȸ�� ��Ʈ���� ���
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);

    // GPU���� ȸ�� ����
    cv::cuda::GpuMat gpuRotatedImage;
    cv::cuda::warpAffine(gpuImage, gpuRotatedImage, rotationMatrix, gpuImage.size());

    // ��� �̹����� CPU �޸𸮷� �ٿ�ε�
    cv::Mat outputImage;
    gpuRotatedImage.download(outputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "rotate", "CUDA", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::rotateCUDAKernel(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    cv::Mat outputImage;
    callRotateImageCUDA(inputImage, outputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "rotate", "CUDAKernel", elapsedTimeMs);

    return result;
}

double ImageProcessor::getCurrentTimeMs()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::steady_clock::now().time_since_epoch()).count();
}

QFuture<bool> ImageProcessor::gaussianBlur(cv::Mat& imageOpenCV
                                            , cv::Mat& imageIPP
                                            , cv::Mat& imageCUDA
                                            , cv::Mat& imageCUDAKernel
                                            , int kernelSize)
{
    //�Լ� �̸��� ���ڿ��� ����
    const char* functionName = __func__;

    return QtConcurrent::run([this
        , &imageOpenCV
        , &imageIPP
        , &imageCUDA
        , &imageCUDAKernel
        , kernelSize
        , functionName]() -> bool {

        QMutexLocker locker(&mutex);

        try {

            if (imageOpenCV.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            if (kernelSize % 2 == 0 || kernelSize < 1) {
                qDebug() << "Invalid kernel size for Gaussian blur.";
                return false;
            }

            pushToUndoStackOpenCV(imageOpenCV.clone());
            pushToUndoStackIPP(imageIPP.clone());
            pushToUndoStackCUDA(imageCUDA.clone());
            pushToUndoStackCUDAKernel(imageCUDAKernel.clone());

            QVector<ProcessingResult> results;

            ProcessingResult outputOpenCV = gaussianBlurOpenCV(imageOpenCV, kernelSize);
            lastProcessedImageOpenCV = outputOpenCV.processedImage.clone();
            results.append(outputOpenCV);

            ProcessingResult outputIPP = gaussianBlurIPP(imageIPP, kernelSize);
            lastProcessedImageIPP = outputIPP.processedImage.clone();
            results.append(outputIPP);

            ProcessingResult outputCUDA = gaussianBlurCUDA(imageCUDA, kernelSize);
            lastProcessedImageCUDA = outputCUDA.processedImage.clone();
            results.append(outputCUDA);

            ProcessingResult outputCUDAKernel = gaussianBlurCUDAKernel(imageCUDAKernel, kernelSize);
            lastProcessedImageCUDAKernel = outputCUDAKernel.processedImage.clone();
            results.append(outputCUDAKernel);

            emit imageProcessed(results);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "Exception occurred while applying Gaussian blur:"
                << e.what();
            return false;
        }
        });
}

ImageProcessor::ProcessingResult ImageProcessor::gaussianBlurOpenCV(cv::Mat& inputImage, int kernelSize)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    ImageProcessorOpenCV IPOpenCV;
    cv::Mat outputImage = IPOpenCV.gaussianBlur(inputImage, kernelSize, 0, 0, 1);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "gaussianBlur", "OpenCV", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::gaussianBlurIPP(cv::Mat& inputImage, int kernelSize) {
    
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    ImageProcessorIPP IPIPP;
    cv::Mat outputImage = IPIPP.gaussianBlur(inputImage, kernelSize);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "gaussianBlur", "IPP", elapsedTimeMs);
    
    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::gaussianBlurCUDA(cv::Mat& inputImage, int kernelSize)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    cv::cuda::GpuMat gpuImage;
    gpuImage.upload(inputImage);

    cv::Ptr<cv::cuda::Filter> gaussianFilter =
    cv::cuda::createGaussianFilter(gpuImage.type()
        , gpuImage.type()
        , cv::Size(kernelSize, kernelSize)
        , 0);

    cv::cuda::GpuMat blurredGpuImage;
    gaussianFilter->apply(gpuImage, blurredGpuImage);

    cv::Mat outputImage;
    blurredGpuImage.download(outputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "gaussianBlur", "CUDA", elapsedTimeMs);

    return result;

}

ImageProcessor::ProcessingResult ImageProcessor::gaussianBlurCUDAKernel(cv::Mat& inputImage, int kernelSize)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    cv::Mat outputImage;
    callGaussianBlurCUDA(inputImage, outputImage, kernelSize);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "gaussianBlur", "CUDAKernel", elapsedTimeMs);

    return result;
}

//Canny
QFuture<bool> ImageProcessor::cannyEdges(cv::Mat& imageOpenCV
                                        , cv::Mat& imageIPP
                                        , cv::Mat& imageCUDA
                                        , cv::Mat& imageCUDAKernel)
{
    //�Լ� �̸��� ���ڿ��� ����
    const char* functionName = __func__;

    return QtConcurrent::run([this
        , &imageOpenCV
        , &imageIPP
        , &imageCUDA
        , &imageCUDAKernel
        , functionName]() -> bool {

        QMutexLocker locker(&mutex);

        try {

            if (imageOpenCV.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            pushToUndoStackOpenCV(imageOpenCV.clone());
            pushToUndoStackIPP(imageIPP.clone());
            pushToUndoStackCUDA(imageCUDA.clone());
            pushToUndoStackCUDAKernel(imageCUDAKernel.clone());            

            QVector<ProcessingResult> results;

            ProcessingResult outputOpenCV = cannyEdgesOpenCV(imageOpenCV);
            lastProcessedImageOpenCV = outputOpenCV.processedImage.clone();
            results.append(outputOpenCV);

            ProcessingResult outputIPP = cannyEdgesIPP(imageIPP);
            lastProcessedImageIPP = outputIPP.processedImage.clone();
            results.append(outputIPP);

            ProcessingResult outputCUDA = cannyEdgesCUDA(imageCUDA);
            lastProcessedImageCUDA = outputCUDA.processedImage.clone();
            results.append(outputCUDA);

            ProcessingResult outputCUDAKernel = cannyEdgesCUDAKernel(imageCUDAKernel);
            lastProcessedImageCUDAKernel = outputCUDAKernel.processedImage.clone();
            results.append(outputCUDAKernel);            

            emit imageProcessed(results);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "Exception occurred while applying Canny edges:" << e.what();
            return false;
        }
        });
}

ImageProcessor::ProcessingResult ImageProcessor::cannyEdgesOpenCV(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    ImageProcessorOpenCV IPOpenCV;
    cv::Mat outputImage = IPOpenCV.cannyEdges(inputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "cannyEdges", "OpenCV", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::cannyEdgesIPP(cv::Mat& inputImage) {
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    ImageProcessorIPP IPIPP;
    cv::Mat outputImage = IPIPP.cannyEdges(inputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // ��� �ð� ���

    result = setResult(result, inputImage, outputImage, "cannyEdges", "IPP", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::cannyEdgesCUDA(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    cv::Mat grayImage;
    if (inputImage.channels() == 3) {
        //grayImage = convertToGrayCUDA(inputImage);
        cv::cuda::GpuMat d_inputImage;
        d_inputImage.upload(inputImage);
        cv::cuda::GpuMat d_outputImage;
        d_outputImage = convertToGrayCUDA(d_inputImage);
        d_outputImage.download(grayImage);
    }
    else {
        grayImage = inputImage.clone();
    }

    // GPU���� ĳ�� ���� ������ ����
    cv::cuda::GpuMat d_gray;
    d_gray.upload(grayImage);

    cv::cuda::GpuMat d_cannyEdges;
    cv::Ptr<cv::cuda::CannyEdgeDetector> cannyDetector = cv::cuda::createCannyEdgeDetector(50, 150);
    cannyDetector->detect(d_gray, d_cannyEdges);

    // ����� CPU �޸𸮷� ����
    cv::Mat edges;
    d_cannyEdges.download(edges);

    // ��� �̹����� �ʷϻ� ���� ǥ��
    cv::Mat outputImage = cv::Mat::zeros(inputImage.size(), CV_8UC3); // 3-channel BGR image
    cv::Mat mask(edges.size(), CV_8UC1, cv::Scalar(0)); // Mask for green edges
    mask.setTo(cv::Scalar(255), edges); // Set pixels to 255 (white) where edges are detected
    cv::Mat channels[3];
    cv::split(outputImage, channels);
    //channels[1] = mask; // Green channel is set by mask
    channels[0] = mask; // Blue channel is set by mask
    channels[1] = mask; // Green channel is set by mask
    channels[2] = mask; // Red channel is set by mask
    cv::merge(channels, 3, outputImage); // Merge channels to get green edges

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "cannyEdges", "CUDA", elapsedTimeMs);

    // GPU �޸� ������ GpuMat ��ü�� �������� ��� �� �ڵ����� ó���˴ϴ�.

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::cannyEdgesCUDAKernel(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    cv::Mat grayImage;
    if (inputImage.channels() == 3) {
        grayImage = convertToGrayCUDAKernel(inputImage);
    }
    else {
        grayImage = inputImage.clone();
    }

    cv::Mat outputImage;
    callCannyEdgesCUDA(grayImage, outputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, grayImage, outputImage, "cannyEdges", "CUDAKernel", elapsedTimeMs);

    return result;
}

QFuture<bool> ImageProcessor::medianFilter(cv::Mat& imageOpenCV
    , cv::Mat& imageIPP
    , cv::Mat& imageCUDA
    , cv::Mat& imageCUDAKernel)
{

    //�Լ� �̸��� ���ڿ��� ����
    const char* functionName = __func__;

    return QtConcurrent::run([this
        , &imageOpenCV
        , &imageIPP
        , &imageCUDA
        , &imageCUDAKernel
        , functionName]() -> bool {

        QMutexLocker locker(&mutex);

        try {

            if (imageOpenCV.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            pushToUndoStackOpenCV(imageOpenCV.clone());
            pushToUndoStackIPP(imageIPP.clone());
            pushToUndoStackCUDA(imageCUDA.clone());
            pushToUndoStackCUDAKernel(imageCUDAKernel.clone());

            QVector<ProcessingResult> results;

            ProcessingResult outputOpenCV = medianFilterOpenCV(imageOpenCV);
            lastProcessedImageOpenCV = outputOpenCV.processedImage.clone();
            results.append(outputOpenCV);

            ProcessingResult outputIPP = medianFilterIPP(imageIPP);
            lastProcessedImageIPP = outputIPP.processedImage.clone();
            results.append(outputIPP);

            ProcessingResult outputCUDA = medianFilterCUDA(imageCUDA);
            lastProcessedImageCUDA = outputCUDA.processedImage.clone();
            results.append(outputCUDA);

            ProcessingResult outputCUDAKernel = medianFilterCUDAKernel(imageCUDAKernel);
            lastProcessedImageCUDAKernel = outputCUDAKernel.processedImage.clone();
            results.append(outputCUDAKernel);

            emit imageProcessed(results);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "median ���� ���� �� ���� �߻�: "
                << e.what();
            return false;
        }
        });
}

ImageProcessor::ProcessingResult ImageProcessor::medianFilterOpenCV(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    ImageProcessorOpenCV IPOpenCV;
    cv::Mat outputImage = IPOpenCV.medianFilter(inputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "medianFilter", "OpenCV", elapsedTimeMs);

    return result;
}

Ipp8u* ImageProcessor::matToIpp8u(cv::Mat& mat)
{
    return mat.ptr<Ipp8u>();
}

ImageProcessor::ProcessingResult ImageProcessor::medianFilterIPP(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    ImageProcessorIPP IPIPP;
    cv::Mat outputImage = IPIPP.medianFilter(inputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "medianFilter", "IPP", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::medianFilterCUDA(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    // Upload image to GPU
    cv::cuda::GpuMat gpuImage;
    gpuImage.upload(inputImage);

    // Create median filter
    cv::Ptr<cv::cuda::Filter> medianFilter =
        cv::cuda::createMedianFilter(gpuImage.type(), 5);

    // Apply median filter on GPU
    cv::cuda::GpuMat medianedGpuImage;
    medianFilter->apply(gpuImage, medianedGpuImage);

    // Download the result back to CPU
    cv::Mat outputImage;
    medianedGpuImage.download(outputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "medianFilter", "CUDA", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::medianFilterCUDAKernel(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    cv::Mat outputImage;
    callMedianFilterCUDA(inputImage, outputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "medianFilter", "OpenCV", elapsedTimeMs);

    return result;
}

QFuture<bool> ImageProcessor::laplacianFilter(cv::Mat& imageOpenCV
    , cv::Mat& imageIPP
    , cv::Mat& imageCUDA
    , cv::Mat& imageCUDAKernel)
{
    //�Լ� �̸��� ���ڿ��� ����
    const char* functionName = __func__;

    return QtConcurrent::run([this
        , &imageOpenCV
        , &imageIPP
        , &imageCUDA
        , &imageCUDAKernel
        , functionName]() -> bool {

        QMutexLocker locker(&mutex);

        try {

            if (imageOpenCV.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            pushToUndoStackOpenCV(imageOpenCV.clone());
            pushToUndoStackIPP(imageIPP.clone());
            pushToUndoStackCUDA(imageCUDA.clone());
            pushToUndoStackCUDAKernel(imageCUDAKernel.clone());

            QVector<ProcessingResult> results;

            ProcessingResult outputOpenCV = laplacianFilterOpenCV(imageOpenCV);
            lastProcessedImageOpenCV = outputOpenCV.processedImage.clone();
            results.append(outputOpenCV);

            ProcessingResult outputIPP = laplacianFilterIPP(imageOpenCV);
            lastProcessedImageIPP = outputIPP.processedImage.clone();
            results.append(outputIPP);

            ProcessingResult outputCUDA = laplacianFilterCUDA(imageCUDA);
            lastProcessedImageCUDA = outputCUDA.processedImage.clone();
            results.append(outputCUDA);

            ProcessingResult outputCUDAKernel = laplacianFilterCUDAKernel(imageCUDAKernel);
            lastProcessedImageCUDAKernel = outputCUDAKernel.processedImage.clone();
            results.append(outputCUDAKernel);                  

            emit imageProcessed(results);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "laplacian ���� ���� �� ���� �߻�: "
                << e.what();
            return false;
        }
        });
}

ImageProcessor::ProcessingResult ImageProcessor::laplacianFilterOpenCV(cv::Mat& inputImage)
{
    std::cout << "laplacianFilter OpenCV" << std::endl;

    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    ImageProcessorOpenCV IPOpenCV;
    cv::Mat outputImage = IPOpenCV.laplacianFilter(inputImage);    

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "laplacianFilter", "OpenCV", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::laplacianFilterIPP(cv::Mat& inputImage)
{
    std::cout << "laplacianFilter IPP" << std::endl;

    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    ImageProcessorIPP IPIPP;
    cv::Mat outputImage = IPIPP.laplacianFilter(inputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "laplacianFilter", "IPP", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::laplacianFilterCUDA(cv::Mat& inputImage)
{
    std::cout << "laplacianFilter CUDA" << std::endl;

    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    // �Է� �̹����� GPU �޸𸮷� ���ε�
    cv::cuda::GpuMat d_inputImage;
    d_inputImage.upload(inputImage);

    // �Է� �̹��� Ÿ�� Ȯ�� �� ä�� �� ��ȯ
    int inputType = d_inputImage.type();
    int depth = CV_MAT_DEPTH(inputType);

    // CUDA Laplacian ���͸� ������ �� �ִ� ������ Ÿ������ ��ȯ
    cv::cuda::GpuMat d_grayImage;
    if (depth != CV_8U && depth != CV_16U && depth != CV_32F) {
        d_inputImage.convertTo(d_grayImage, CV_32F);  // �Է� �̹����� CV_32F�� ��ȯ
    }
    else if (d_inputImage.channels() == 3) {
        cv::cuda::cvtColor(d_inputImage, d_grayImage, cv::COLOR_BGR2GRAY); // RGB �̹����� grayscale�� ��ȯ
    }
    else {
        d_grayImage = d_inputImage.clone();  // �̹� ������ Ÿ���� ��� �״�� ���
    }

    // Laplacian ���͸� ������ �� �Է� �� ��� �̹��� Ÿ���� �����ϰ� ����
    int srcType = d_grayImage.type();
    cv::Ptr<cv::cuda::Filter> laplacianFilter = cv::cuda::createLaplacianFilter(srcType, srcType, 3);

    // ��� �̹��� �޸� �Ҵ�
    cv::cuda::GpuMat d_outputImage(d_grayImage.size(), srcType);

    // Laplacian ���� ����
    laplacianFilter->apply(d_grayImage, d_outputImage);

    // GPU���� CPU�� ��� �̹��� �ٿ�ε�
    cv::Mat outputImage;
    d_outputImage.download(outputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "laplacianFilter", "CUDA", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::laplacianFilterCUDAKernel(cv::Mat& inputImage)
{
    std::cout << "laplacianFilter CUDAKernel" << std::endl;

    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    cv::Mat outputImage;
    callLaplacianFilterCUDA(inputImage, outputImage);     
    // outputImage�� ����Ͽ� ���� Ȯ��
    //std::cout << "Output Image:" << std::endl;
    //std::cout << outputImage << std::endl;

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "laplacianFilter", "CUDAKernel", elapsedTimeMs);

    return result;
}


QFuture<bool> ImageProcessor::bilateralFilter(cv::Mat& imageOpenCV
                                            , cv::Mat& imageIPP
                                            , cv::Mat& imageCUDA
                                            , cv::Mat& imageCUDAKernel)
{
    //�Լ� �̸��� ���ڿ��� ����
    const char* functionName = __func__;

    return QtConcurrent::run([this
        , &imageOpenCV
        , &imageIPP
        , &imageCUDA
        , &imageCUDAKernel
        , functionName]() -> bool {

        QMutexLocker locker(&mutex);

        try {

            if (imageOpenCV.empty()) {
                qDebug() << "Input image is empty.";
                return false;
            }

            pushToUndoStackOpenCV(imageOpenCV.clone());
            pushToUndoStackIPP(imageIPP.clone());
            pushToUndoStackCUDA(imageCUDA.clone());
            pushToUndoStackCUDAKernel(imageCUDAKernel.clone());

            QVector<ProcessingResult> results;

            ProcessingResult outputOpenCV = bilateralFilterOpenCV(imageOpenCV);
            lastProcessedImageOpenCV = outputOpenCV.processedImage.clone();
            results.append(outputOpenCV);

            ProcessingResult outputIPP = bilateralFilterIPP(imageIPP);
            lastProcessedImageIPP = outputIPP.processedImage.clone();
            results.append(outputIPP);

            ProcessingResult outputCUDA = bilateralFilterCUDA(imageCUDA);
            lastProcessedImageCUDA = outputCUDA.processedImage.clone();
            results.append(outputCUDA);

            ProcessingResult outputCUDAKernel = bilateralFilterCUDAKernel(imageIPP);
            lastProcessedImageCUDAKernel = outputCUDAKernel.processedImage.clone();
            results.append(outputCUDAKernel);   

            emit imageProcessed(results);

            return true;
        }
        catch (const cv::Exception& e) {
            qDebug() << "bilateral ���� ���� �� ���� �߻�: "
                << e.what();
            return false;
        }
        });
}

ImageProcessor::ProcessingResult ImageProcessor::bilateralFilterOpenCV(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    ImageProcessorOpenCV IPOpenCV;
    cv::Mat outputImage = IPOpenCV.bilateralFilter(inputImage);   

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "bilateralFilter", "OpenCV", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::bilateralFilterIPP(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    ImageProcessorIPP IPIPP;
    cv::Mat outputImage = IPIPP.bilateralFilter(inputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "bilateralFilter", "IPP", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::bilateralFilterCUDA(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    //cv::cuda ���� createBilateralFilter �������� �ʾ� CUDA Kernel�� �ؾ���
    cv::Mat outputImage;
    callBilateralFilterCUDA(inputImage, outputImage, 9, 75, 75);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "bilateralFilter", "CUDA-not support", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::bilateralFilterCUDAKernel(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    cv::Mat outputImage;
    callBilateralFilterCUDA(inputImage, outputImage, 9, 75, 75);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "bilateralFilter", "CUDAKernel", elapsedTimeMs);

    return result;
}

QFuture<bool> ImageProcessor::sobelFilter(cv::Mat& imageOpenCV
                                        , cv::Mat& imageIPP
                                        , cv::Mat& imageCUDA
                                        , cv::Mat& imageCUDAKernel)
{
    //�Լ� �̸��� ���ڿ��� ����
    const char* functionName = __func__;

    return QtConcurrent::run([this
        , &imageOpenCV
        , &imageIPP
        , &imageCUDA
        , &imageCUDAKernel
        , functionName]() -> bool {

        if (imageOpenCV.empty()) {
            qDebug() << "Input image is empty.";
            return false;
        }

        pushToUndoStackOpenCV(imageOpenCV.clone());
        pushToUndoStackIPP(imageIPP.clone());
        pushToUndoStackCUDA(imageCUDA.clone());
        pushToUndoStackCUDAKernel(imageCUDAKernel.clone());

        QVector<ProcessingResult> results;

        ProcessingResult outputOpenCV = sobelFilterOpenCV(imageOpenCV);
        lastProcessedImageOpenCV = outputOpenCV.processedImage.clone();
        results.append(outputOpenCV);

        ProcessingResult outputIPP = sobelFilterIPP(imageIPP);
        lastProcessedImageIPP = outputIPP.processedImage.clone();
        results.append(outputIPP);

        ProcessingResult outputCUDA = sobelFilterCUDA(imageCUDA);
        lastProcessedImageCUDA = outputCUDA.processedImage.clone();
        results.append(outputCUDA);

        ProcessingResult outputCUDAKernel = sobelFilterCUDAKernel(imageCUDAKernel);
        lastProcessedImageCUDAKernel = outputCUDAKernel.processedImage.clone();
        results.append(outputCUDAKernel);

        emit imageProcessed(results);

        return true;
        });
}

ImageProcessor::ProcessingResult ImageProcessor::sobelFilterOpenCV(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    ImageProcessorOpenCV IPOpenCV;
    cv::Mat outputImage = IPOpenCV.sobelFilter(inputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "sobelFilter", "OpenCV", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::sobelFilterIPP(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    ImageProcessorIPP IPIPP;
    cv::Mat outputImage = IPIPP.sobelFilter(inputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "sobelFilter", "IPP", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::sobelFilterCUDA(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    cv::Mat outputImage;

    // Convert input image to grayscale if necessary
    cv::Mat grayImage;
    if (inputImage.channels() > 1) {
        cv::cuda::GpuMat d_inputImage;
        d_inputImage.upload(inputImage);
        cv::cuda::GpuMat d_outputImage;
        d_outputImage = convertToGrayCUDA(d_inputImage);
        d_outputImage.download(grayImage);
    }
    else {
        grayImage = inputImage;
    }

    // Transfer input image to GPU
    cv::cuda::GpuMat d_inputImage(grayImage);
    cv::cuda::GpuMat d_outputImage;

    // Create Sobel filter on GPU
    cv::Ptr<cv::cuda::Filter> sobelFilter = cv::cuda::createSobelFilter(
        d_inputImage.type(),   // srcType
        CV_16SC1,              // dstType
        1,                     // dx (order of derivative in x)
        0,                     // dy (order of derivative in y)
        3                      // ksize (kernel size, 3x3 Sobel)
    );

    // Apply Sobel filter on GPU
    sobelFilter->apply(d_inputImage, d_outputImage);

    // Transfer result back to CPU
    d_outputImage.download(outputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "sobelFilter", "CUDA", elapsedTimeMs);

    return result;
}

ImageProcessor::ProcessingResult ImageProcessor::sobelFilterCUDAKernel(cv::Mat& inputImage)
{
    ProcessingResult result;
    double startTime = cv::getTickCount(); // ���� �ð� ����

    cv::Mat outputImage = convertToGrayCUDAKernel(inputImage);
    callSobelFilterCUDA(inputImage, outputImage);

    double endTime = cv::getTickCount(); // ���� �ð� ����
    double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

    result = setResult(result, inputImage, outputImage, "sobelFilter", "CUDAKernel", elapsedTimeMs);

    return result;
}

bool ImageProcessor::canUndoOpenCV() const
{
    return !undoStackOpenCV.empty();
}

bool ImageProcessor::canRedoOpenCV() const
{
    return !redoStackOpenCV.empty();
}

//�������
// Undo operation
void ImageProcessor::undo()
{
    const char* functionName = __func__;
    QVector<ProcessingResult> results;

    try {
        if (!canUndoOpenCV()) {
            throw std::runtime_error("Cannot undo: Undo stack is empty");
        }

        double startTime = cv::getTickCount(); // ���� �ð� ����        

        // ���� �̹����� redo ���ÿ� Ǫ��
        redoStackOpenCV.push(lastProcessedImageOpenCV.clone());
        redoStackIPP.push(lastProcessedImageIPP.clone());
        redoStackCUDA.push(lastProcessedImageCUDA.clone());
        redoStackCUDAKernel.push(lastProcessedImageCUDAKernel.clone());

        // undo ���ÿ��� �̹��� ����
        lastProcessedImageOpenCV = undoStackOpenCV.top().clone();
        lastProcessedImageIPP = undoStackIPP.top().clone();
        lastProcessedImageCUDA = undoStackCUDA.top().clone();
        lastProcessedImageCUDAKernel = undoStackCUDAKernel.top().clone();

        // undo ���ÿ��� �̹��� ����
        undoStackOpenCV.pop();
        undoStackIPP.pop();
        undoStackCUDA.pop();
        undoStackCUDAKernel.pop();

        QString outputInfoOpenCV = "(Output) Channels: " + QString::number(lastProcessedImageOpenCV.channels())
            + ", type: " + QString::number(lastProcessedImageOpenCV.type())
            + ", depth: " + QString::number(lastProcessedImageOpenCV.depth());
        QString outputInfoIPP = "(Output) Channels: " + QString::number(lastProcessedImageIPP.channels())
            + ", type: " + QString::number(lastProcessedImageIPP.type())
            + ", depth: " + QString::number(lastProcessedImageIPP.depth());
        QString outputInfoCUDA = "(Output) Channels: " + QString::number(lastProcessedImageCUDA.channels())
            + ", type: " + QString::number(lastProcessedImageCUDA.type())
            + ", depth: " + QString::number(lastProcessedImageCUDA.depth());
        QString outputInfoCUDAKernel = "(Output) Channels: " + QString::number(lastProcessedImageCUDAKernel.channels())
            + ", type: " + QString::number(lastProcessedImageCUDAKernel.type())
            + ", depth: " + QString::number(lastProcessedImageCUDAKernel.depth());

        double endTime = cv::getTickCount(); // ���� �ð� ����
        double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

        // ��� ����
        results.append(ProcessingResult(functionName, "OpenCV", lastProcessedImageOpenCV.clone(), elapsedTimeMs, "", outputInfoOpenCV));
        results.append(ProcessingResult(functionName, "IPP", lastProcessedImageIPP.clone(), elapsedTimeMs, "", outputInfoIPP));
        results.append(ProcessingResult(functionName, "CUDA", lastProcessedImageCUDA.clone(), elapsedTimeMs, "", outputInfoCUDA));
        results.append(ProcessingResult(functionName, "CUDAKernel", lastProcessedImageCUDAKernel.clone(), elapsedTimeMs, "", outputInfoCUDAKernel));

        emit imageProcessed(results);
    }
    catch (const std::exception& e) {
        qDebug() << "Exception occurred in ImageProcessor::undo(): " << e.what();
    }
}


//�����
void ImageProcessor::redo()
{
    const char* functionName = __func__;
    QVector<ProcessingResult> results;

    try {
        if (!canRedoOpenCV()) {
            throw std::runtime_error("Cannot redo: Redo stack is empty");
        }

        double startTime = cv::getTickCount(); // ���� �ð� ����        

        // ���� �̹����� undo ���ÿ� Ǫ��
        undoStackOpenCV.push(lastProcessedImageOpenCV.clone());
        undoStackIPP.push(lastProcessedImageIPP.clone());
        undoStackCUDA.push(lastProcessedImageCUDA.clone());
        undoStackCUDAKernel.push(lastProcessedImageCUDAKernel.clone());

        // redo ���ÿ��� �̹��� ����
        lastProcessedImageOpenCV = redoStackOpenCV.top().clone();
        lastProcessedImageIPP = redoStackIPP.top().clone();
        lastProcessedImageCUDA = redoStackCUDA.top().clone();
        lastProcessedImageCUDAKernel = redoStackCUDAKernel.top().clone();

        // redo ���ÿ��� �̹��� ����
        redoStackOpenCV.pop();
        redoStackIPP.pop();
        redoStackCUDA.pop();
        redoStackCUDAKernel.pop();

        QString outputInfoOpenCV = "(Output) Channels: " + QString::number(lastProcessedImageOpenCV.channels())
            + ", type: " + QString::number(lastProcessedImageOpenCV.type())
            + ", depth: " + QString::number(lastProcessedImageOpenCV.depth());
        QString outputInfoIPP = "(Output) Channels: " + QString::number(lastProcessedImageIPP.channels())
            + ", type: " + QString::number(lastProcessedImageIPP.type())
            + ", depth: " + QString::number(lastProcessedImageIPP.depth());
        QString outputInfoCUDA = "(Output) Channels: " + QString::number(lastProcessedImageCUDA.channels())
            + ", type: " + QString::number(lastProcessedImageCUDA.type())
            + ", depth: " + QString::number(lastProcessedImageCUDA.depth());
        QString outputInfoCUDAKernel = "(Output) Channels: " + QString::number(lastProcessedImageCUDAKernel.channels())
            + ", type: " + QString::number(lastProcessedImageCUDAKernel.type())
            + ", depth: " + QString::number(lastProcessedImageCUDAKernel.depth());

        double endTime = cv::getTickCount(); // ���� �ð� ����
        double elapsedTimeMs = (endTime - startTime) / cv::getTickFrequency() * 1000.0; // �ð� ���

        // ��� ����
        results.append(ProcessingResult(functionName, "OpenCV", lastProcessedImageOpenCV.clone(), elapsedTimeMs, "", outputInfoOpenCV));
        results.append(ProcessingResult(functionName, "IPP", lastProcessedImageIPP.clone(), elapsedTimeMs, "", outputInfoIPP));
        results.append(ProcessingResult(functionName, "CUDA", lastProcessedImageCUDA.clone(), elapsedTimeMs, "", outputInfoCUDA));
        results.append(ProcessingResult(functionName, "CUDAKernel", lastProcessedImageCUDAKernel.clone(), elapsedTimeMs, "", outputInfoCUDAKernel));

        emit imageProcessed(results);
    }
    catch (const std::exception& e) {
        qDebug() << "Exception occurred in ImageProcessor::redo(): " << e.what();
    }
}



void ImageProcessor::cleanUndoStack()
{
    QMutexLocker locker(&mutex);
    while (!undoStackOpenCV.empty()) {
        undoStackOpenCV.pop();
    }

    while (!undoStackIPP.empty()) {
        undoStackIPP.pop();
    }

    while (!undoStackCUDA.empty()) {
        undoStackCUDA.pop();
    }

    while (!undoStackCUDAKernel.empty()) {
        undoStackCUDAKernel.pop();
    }
}

void ImageProcessor::cleanRedoStack()
{
    QMutexLocker locker(&mutex);
    while (!redoStackOpenCV.empty()) {
        redoStackOpenCV.pop();
    }

    while (!redoStackIPP.empty()) {
        redoStackIPP.pop();
    }

    while (!redoStackCUDA.empty()) {
        redoStackCUDA.pop();
    }

    while (!redoStackCUDAKernel.empty()) {
        redoStackCUDAKernel.pop();
    }
}

void ImageProcessor::initializeCUDA()
{
    // ������ ���� �۾��� �����Ͽ� CUDA �ʱ�ȭ�� ����
    cv::cuda::GpuMat temp;
    temp.upload(cv::Mat::zeros(1, 1, CV_8UC1));
    cv::cuda::cvtColor(temp, temp, cv::COLOR_GRAY2BGR);
}

const cv::Mat& ImageProcessor::getLastProcessedImageOpenCV() const
{
    return lastProcessedImageOpenCV;
}

const cv::Mat& ImageProcessor::getLastProcessedImageIPP() const
{
    return lastProcessedImageIPP;
}

const cv::Mat& ImageProcessor::getLastProcessedImageCUDA() const
{
    return lastProcessedImageCUDA;
}

const cv::Mat& ImageProcessor::getLastProcessedImageCUDAKernel() const
{
    return lastProcessedImageCUDAKernel;
}

void ImageProcessor::pushToUndoStackOpenCV(const cv::Mat& image)
{
    undoStackOpenCV.push(image.clone());
}

void ImageProcessor::pushToUndoStackIPP(const cv::Mat& image)
{
    undoStackIPP.push(image.clone());
}

void ImageProcessor::pushToUndoStackCUDA(const cv::Mat& image)
{
    undoStackCUDA.push(image.clone());
}

void ImageProcessor::pushToUndoStackCUDAKernel(const cv::Mat& image)
{
    undoStackCUDAKernel.push(image.clone());
}

void ImageProcessor::pushToRedoStackOpenCV(const cv::Mat& image)
{
    redoStackOpenCV.push(image.clone());
}

void ImageProcessor::pushToRedoStackIPP(const cv::Mat& image)
{
    redoStackIPP.push(image.clone());
}

void ImageProcessor::pushToRedoStackCUDA(const cv::Mat& image)
{
    redoStackCUDA.push(image.clone());
}

void ImageProcessor::pushToRedoStackCUDAKernel(const cv::Mat& image)
{
    redoStackCUDAKernel.push(image.clone());
}

ImageProcessor::ProcessingResult ImageProcessor::setResult(ProcessingResult& result, cv::Mat& inputImage, cv::Mat& outputImage, QString functionName, QString processName, double processingTime)
{
    result.functionName = functionName;
    result.processName = processName;
    result.inputInfo = "\n(Input) Channels: " + QString::number(inputImage.channels())
        + ", type: " + QString::number(inputImage.type())
        + "(" + ImageTypeConverter::getImageTypeString(inputImage.type()) + ")"
        + ", depth: " + QString::number(inputImage.depth());
    result.processedImage = outputImage.clone();
    result.processingTime = processingTime;
    result.outputInfo = "\n(Output) Channels: " + QString::number(outputImage.channels())
        + ", type: " + QString::number(outputImage.type())
        + "(" + ImageTypeConverter::getImageTypeString(outputImage.type()) + ")"
        + ", depth: " + QString::number(outputImage.depth());

    return result;
}
