//MainWindow.h
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QImage>
#include <QPixmap>
#include <QFileDialog>
#include <QMessageBox>
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

private:
    Ui::MainWindow* ui;
    cv::Mat currentImage;
    ImageProcessor* imageProcessor;
    double scaleFactor;

    void displayImage(const cv::Mat& image);
};

#endif // MAINWINDOW_H
