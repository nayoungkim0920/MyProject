#ifndef YOLOV5DIALOG_H
#define YOLOV5DIALOG_H

#include <QDialog>
#include <opencv2/opencv.hpp>
#include "ui_yolov5_result.h"
#include "MainWindow.h"

class YOLOv5Dialog : public QDialog
{
    Q_OBJECT

public:
    explicit YOLOv5Dialog(QWidget* parent = nullptr);
    ~YOLOv5Dialog();

    void setImages(const cv::Mat& imageOpenCV
                    , const cv::Mat& imageIPP
                    , const cv::Mat& imageNPP
                    , const cv::Mat& imageCUDA
                    , const cv::Mat& imageCUDAKernel
                    , const cv::Mat& imageGStreamer);

private:
    Ui::YOLOv5Dialog* ui;

    cv::Mat currentImageOpenCV;
    cv::Mat currentImageIPP;
    cv::Mat currentImageCUDA;
    cv::Mat currentImageCUDAKernel;
    cv::Mat currentImageNPP;
    cv::Mat currentImageGStreamer;

    void displayImage(cv::Mat image, QLabel* label);
};

#endif // YOLOV5DIALOG_H