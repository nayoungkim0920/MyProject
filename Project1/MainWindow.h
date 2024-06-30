//MainWindow.h
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QImage>
#include <QPixmap>
#include <QFileDialog>
#include <QMessageBox>
#include <QInputDialog>
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
    void convertToGrayscale();
    void applyGaussianBlur();
    void cannyEdges();
    void medianFilter();
    void laplacianFilter();
    void bilateralFilter();
    void exitApplication();
    void redoAction();
    void undoAction();
    void first();
    void displayImage(const cv::Mat& image); 

private:
    Ui::MainWindow* ui;
    cv::Mat currentImage;
    ImageProcessor* imageProcessor;
    double scaleFactor;
    cv::Mat initialImage;
    cv::Mat previousImage;
    
    void connectActions();
    void connectImageProcessor();
    void setInitialWindowGeometry();

    template<typename Func, typename... Args>
    inline void applyImageProcessing(Func func, Args&&... args);
};

#endif // MAINWINDOW_H

