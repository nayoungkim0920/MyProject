/********************************************************************************
** Form generated from reading UI file 'yolov5_result.ui'
**
** Created by: Qt User Interface Compiler version 6.7.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_YOLOV5_RESULT_H
#define UI_YOLOV5_RESULT_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_YOLOv5Dialog
{
public:
    QVBoxLayout *verticalLayout_main;
    QVBoxLayout *verticalLayout_images_and_times;
    QHBoxLayout *horizontalLayout_row1;
    QLabel *label_opencv;
    QLabel *label_ipp;
    QLabel *label_npp;
    QHBoxLayout *horizontalLayout_row2;
    QLabel *label_opencv_time;
    QLabel *label_ipp_time;
    QLabel *label_npp_time;
    QHBoxLayout *horizontalLayout_row3;
    QLabel *label_cuda;
    QLabel *label_cudakernel;
    QLabel *label_gstreamer;
    QHBoxLayout *horizontalLayout_row4;
    QLabel *label_cuda_time;
    QLabel *label_cudakernel_time;
    QLabel *label_gstreamer_time;

    void setupUi(QDialog *YOLOv5Dialog)
    {
        if (YOLOv5Dialog->objectName().isEmpty())
            YOLOv5Dialog->setObjectName("YOLOv5Dialog");
        YOLOv5Dialog->resize(800, 600);
        verticalLayout_main = new QVBoxLayout(YOLOv5Dialog);
        verticalLayout_main->setObjectName("verticalLayout_main");
        verticalLayout_images_and_times = new QVBoxLayout();
        verticalLayout_images_and_times->setObjectName("verticalLayout_images_and_times");
        horizontalLayout_row1 = new QHBoxLayout();
        horizontalLayout_row1->setObjectName("horizontalLayout_row1");
        label_opencv = new QLabel(YOLOv5Dialog);
        label_opencv->setObjectName("label_opencv");
        label_opencv->setAlignment(Qt::AlignCenter);
        label_opencv->setScaledContents(true);
        label_opencv->setFixedSize(QSize(400, 300));

        horizontalLayout_row1->addWidget(label_opencv);

        label_ipp = new QLabel(YOLOv5Dialog);
        label_ipp->setObjectName("label_ipp");
        label_ipp->setAlignment(Qt::AlignCenter);
        label_ipp->setScaledContents(true);
        label_ipp->setFixedSize(QSize(400, 300));

        horizontalLayout_row1->addWidget(label_ipp);

        label_npp = new QLabel(YOLOv5Dialog);
        label_npp->setObjectName("label_npp");
        label_npp->setAlignment(Qt::AlignCenter);
        label_npp->setScaledContents(true);
        label_npp->setFixedSize(QSize(400, 300));

        horizontalLayout_row1->addWidget(label_npp);


        verticalLayout_images_and_times->addLayout(horizontalLayout_row1);

        horizontalLayout_row2 = new QHBoxLayout();
        horizontalLayout_row2->setObjectName("horizontalLayout_row2");
        label_opencv_time = new QLabel(YOLOv5Dialog);
        label_opencv_time->setObjectName("label_opencv_time");
        label_opencv_time->setAlignment(Qt::AlignLeft);

        horizontalLayout_row2->addWidget(label_opencv_time);

        label_ipp_time = new QLabel(YOLOv5Dialog);
        label_ipp_time->setObjectName("label_ipp_time");
        label_ipp_time->setAlignment(Qt::AlignLeft);

        horizontalLayout_row2->addWidget(label_ipp_time);

        label_npp_time = new QLabel(YOLOv5Dialog);
        label_npp_time->setObjectName("label_npp_time");
        label_npp_time->setAlignment(Qt::AlignLeft);

        horizontalLayout_row2->addWidget(label_npp_time);


        verticalLayout_images_and_times->addLayout(horizontalLayout_row2);

        horizontalLayout_row3 = new QHBoxLayout();
        horizontalLayout_row3->setObjectName("horizontalLayout_row3");
        label_cuda = new QLabel(YOLOv5Dialog);
        label_cuda->setObjectName("label_cuda");
        label_cuda->setAlignment(Qt::AlignCenter);
        label_cuda->setScaledContents(true);
        label_cuda->setFixedSize(QSize(400, 300));

        horizontalLayout_row3->addWidget(label_cuda);

        label_cudakernel = new QLabel(YOLOv5Dialog);
        label_cudakernel->setObjectName("label_cudakernel");
        label_cudakernel->setAlignment(Qt::AlignCenter);
        label_cudakernel->setScaledContents(true);
        label_cudakernel->setFixedSize(QSize(400, 300));

        horizontalLayout_row3->addWidget(label_cudakernel);

        label_gstreamer = new QLabel(YOLOv5Dialog);
        label_gstreamer->setObjectName("label_gstreamer");
        label_gstreamer->setAlignment(Qt::AlignCenter);
        label_gstreamer->setScaledContents(true);
        label_gstreamer->setFixedSize(QSize(400, 300));

        horizontalLayout_row3->addWidget(label_gstreamer);


        verticalLayout_images_and_times->addLayout(horizontalLayout_row3);

        horizontalLayout_row4 = new QHBoxLayout();
        horizontalLayout_row4->setObjectName("horizontalLayout_row4");
        label_cuda_time = new QLabel(YOLOv5Dialog);
        label_cuda_time->setObjectName("label_cuda_time");
        label_cuda_time->setAlignment(Qt::AlignLeft);

        horizontalLayout_row4->addWidget(label_cuda_time);

        label_cudakernel_time = new QLabel(YOLOv5Dialog);
        label_cudakernel_time->setObjectName("label_cudakernel_time");
        label_cudakernel_time->setAlignment(Qt::AlignLeft);

        horizontalLayout_row4->addWidget(label_cudakernel_time);

        label_gstreamer_time = new QLabel(YOLOv5Dialog);
        label_gstreamer_time->setObjectName("label_gstreamer_time");
        label_gstreamer_time->setAlignment(Qt::AlignLeft);

        horizontalLayout_row4->addWidget(label_gstreamer_time);


        verticalLayout_images_and_times->addLayout(horizontalLayout_row4);


        verticalLayout_main->addLayout(verticalLayout_images_and_times);


        retranslateUi(YOLOv5Dialog);

        QMetaObject::connectSlotsByName(YOLOv5Dialog);
    } // setupUi

    void retranslateUi(QDialog *YOLOv5Dialog)
    {
        YOLOv5Dialog->setWindowTitle(QCoreApplication::translate("YOLOv5Dialog", "YOLOv5 Detection Results", nullptr));
        label_opencv->setText(QCoreApplication::translate("YOLOv5Dialog", "OpenCV", nullptr));
        label_ipp->setText(QCoreApplication::translate("YOLOv5Dialog", "IPP", nullptr));
        label_npp->setText(QCoreApplication::translate("YOLOv5Dialog", "NPP", nullptr));
        label_opencv_time->setText(QCoreApplication::translate("YOLOv5Dialog", "Processing Time: N/A", nullptr));
        label_ipp_time->setText(QCoreApplication::translate("YOLOv5Dialog", "Processing Time: N/A", nullptr));
        label_npp_time->setText(QCoreApplication::translate("YOLOv5Dialog", "Processing Time: N/A", nullptr));
        label_cuda->setText(QCoreApplication::translate("YOLOv5Dialog", "CUDA", nullptr));
        label_cudakernel->setText(QCoreApplication::translate("YOLOv5Dialog", "CUDA Kernel", nullptr));
        label_gstreamer->setText(QCoreApplication::translate("YOLOv5Dialog", "GStreamer", nullptr));
        label_cuda_time->setText(QCoreApplication::translate("YOLOv5Dialog", "Processing Time: N/A", nullptr));
        label_cudakernel_time->setText(QCoreApplication::translate("YOLOv5Dialog", "Processing Time: N/A", nullptr));
        label_gstreamer_time->setText(QCoreApplication::translate("YOLOv5Dialog", "Processing Time: N/A", nullptr));
    } // retranslateUi

};

namespace Ui {
    class YOLOv5Dialog: public Ui_YOLOv5Dialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_YOLOV5_RESULT_H
