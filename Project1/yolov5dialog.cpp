#include "yolov5dialog.h"

YOLOv5Dialog::YOLOv5Dialog(QWidget* parent)
    : QDialog(parent), ui(new Ui::YOLOv5Dialog)
{
    ui->setupUi(this);
}

YOLOv5Dialog::~YOLOv5Dialog()
{
    delete ui;
}

void YOLOv5Dialog::setImages(const cv::Mat& imageOpenCV, const cv::Mat& imageIPP
    , const cv::Mat& imageNPP, const cv::Mat& imageCUDA, const cv::Mat& imageCUDAKernel
    , const cv::Mat& imageGStreamer)
{
    currentImageOpenCV = imageOpenCV;
    currentImageIPP = imageIPP;
    currentImageNPP = imageNPP;
    currentImageCUDA = imageCUDA;
    currentImageCUDAKernel = imageCUDAKernel;
    currentImageGStreamer = imageGStreamer;

    displayImage(currentImageOpenCV, ui->label_opencv);
    displayImage(currentImageIPP, ui->label_ipp);
    displayImage(currentImageNPP, ui->label_npp);
    displayImage(currentImageCUDA, ui->label_npp);
    displayImage(currentImageCUDAKernel, ui->label_cudakernel);
    displayImage(currentImageGStreamer, ui->label_gstreamer);
}

void YOLOv5Dialog::displayImage(cv::Mat image, QLabel* label)
{
    // 이미지 타입에 따라 QImage를 생성합니다.
    QImage qImage;

    qDebug() << "displayImage() called with image type:" << image.type();

    // OpenCV의 Mat 이미지 타입에 따라 다른 QImage 형식을 사용합니다.
    if (image.type() == CV_8UC1) {
        qDebug() << "displayImage() type: grayscale CV_8UC1 Format_Grayscale8";
        qImage = QImage(image.data,
            image.cols,
            image.rows,
            static_cast<int>(image.step),
            QImage::Format_Grayscale8);
    }
    else if (image.type() == CV_8UC3) {
        qDebug() << ">>displayImage() type: BGR CV_8UC3 Format_RGB888";
        qImage = QImage(image.data,
            image.cols,
            image.rows,
            static_cast<int>(image.step),
            QImage::Format_RGB888).rgbSwapped(); // BGR -> RGB 순서로 변환
    }
    else if (image.type() == CV_8UC4) {
        qDebug() << "displayImage() type: BGRA CV_8UC4 Format_RGBA8888";
        qImage = QImage(image.data,
            image.cols,
            image.rows,
            static_cast<int>(image.step),
            QImage::Format_RGBA8888);
    }
    else if (image.type() == CV_16UC3) {
        qDebug() << "displayImage() type: BGR CV_16UC3 Format_RGB16";

        // 16-bit 이미지를 8-bit로 변환
        cv::Mat temp;
        image.convertTo(temp, CV_8UC3, 1.0 / 256.0);
        qImage = QImage(temp.data,
            temp.cols,
            temp.rows,
            static_cast<int>(temp.step),
            QImage::Format_RGB888).rgbSwapped(); // BGR -> RGB 순서로 변환
    }
    else if (image.type() == CV_16SC1) {
        qDebug() << "displayImage() type: 16-bit signed integer CV_16SC1 Format_Grayscale16";
        qImage = QImage(reinterpret_cast<const uchar*>(image.data),
            image.cols,
            image.rows,
            static_cast<int>(image.step),
            QImage::Format_Grayscale16);
    }
    else if (image.type() == CV_16SC3) {
        qDebug() << "displayImage() type: 16-bit signed integer CV_16SC3 Format_RGB16";
        qImage = QImage(reinterpret_cast<const uchar*>(image.data),
            image.cols,
            image.rows,
            static_cast<int>(image.step),
            QImage::Format_RGB16);
    }
    else {
        qDebug() << "displayImage() type: " << image.type() << " not supported";
        return; // 지원하지 않는 이미지 타입은 처리하지 않음
    }

    // QLabel 위젯에 QPixmap으로 이미지를 설정합니다.
    QPixmap pixmap = QPixmap::fromImage(qImage);
    label->setPixmap(pixmap);
    label->setScaledContents(false); // 이미지를 Label 크기에 맞게 조정
    label->adjustSize(); // Label 크기 조정
    qDebug() << "displayImage() finished";
}
