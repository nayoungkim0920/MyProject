//MainWindow.h
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QImage>
#include <QPixmap>
#include <QFileDialog>
#include <QMessageBox>
#include <QInputDialog>
#include <QGuiApplication>
#include <opencv2/opencv.hpp>
#include "ImageProcessor.h"
#include "ui_MainWindow.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

private slots:
    void openFile();
    void saveFile();
    void rotateImage();
    void zoomInImage();
    void zoomOutImage();
    void grayScale();
    void gaussianBlur();
    void cannyEdges();
    void medianFilter();
    void laplacianFilter();
    void bilateralFilter();
    void sobelFilter();
    void exitApplication();
    void redoAction();
    void undoAction();
    void first();
    void displayImage(cv::Mat image, QLabel* label);
    void handleImageProcessed(QVector<ImageProcessor::ProcessingResult> results);

private:
    Ui::MainWindow* ui;

    cv::Mat currentImage;

    cv::Mat currentImageOpenCV;
    cv::Mat currentImageIPP;
    cv::Mat currentImageCUDA;
    cv::Mat currentImageCUDAKernel;
    cv::Mat currentImageNPP;
    cv::Mat currentImageGStreamer;

    ImageProcessor* imageProcessor;

    double scaleFactor;

    cv::Mat initialImageOpenCV;
    cv::Mat initialImageIPP;
    cv::Mat initialImageCUDA;
    cv::Mat initialImageCUDAKernel;
    cv::Mat initialImageNPP;
    cv::Mat initialImageGStreamer;

    cv::Mat previousImageOpenCV;
    cv::Mat previousImageIPP;
    cv::Mat previousImageCUDA;
    cv::Mat previousImageCUDAKernel;
    cv::Mat previousImageNPP;
    cv::Mat previousImageGStreamer;

    void connectActions();
    void connectImageProcessor();
    void setInitialWindowGeometry();

    //template<typename Func, typename... Args>
    //inline void applyImageProcessing(Func func, Args&&... args);
};

#endif // MAINWINDOW_H
