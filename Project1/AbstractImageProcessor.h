#ifndef ABSTRACTIMAGEPROCESSOR_H
#define ABSTRACTIMAGEPROCESSOR_H

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

// �߻� Ŭ����
class AbstractImageProcessor {
public:
    virtual ~AbstractImageProcessor() {}

    // ���� ���� �Լ� (�ڽ� Ŭ�������� �ݵ�� �����ؾ� ��)
    virtual cv::Mat rotate(cv::Mat& inputImage, bool isRight) = 0;
    virtual cv::Mat grayScale(cv::Mat& inputImage) = 0;
    virtual cv::Mat zoom(cv::Mat& inputImage, int newWidth, int newHeight) = 0;
    virtual cv::Mat gaussianBlur(cv::Mat& inputImage, int kernelSize) = 0;
    virtual cv::Mat cannyEdges(cv::Mat& inputImage) = 0;
    virtual cv::Mat medianFilter(cv::Mat& inputImage) = 0;
    virtual cv::Mat laplacianFilter(cv::Mat& inputImage) = 0;
    virtual cv::Mat bilateralFilter(cv::Mat& inputImage) = 0;
    virtual cv::Mat sobelFilter(cv::Mat& inputImage) = 0;

private:
    virtual std::string getClassName() const = 0;
};

#endif // ABSTRACTIMAGEPROCESSOR_H
