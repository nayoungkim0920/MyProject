//MainWindow.cpp
#include "MainWindow.h"
#include "ui_MainWindow.h"

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    imageProcessor = new ImageProcessor(this);
    scaleFactor = 1.0;

    connect(ui->actionOpen, &QAction::triggered, this, &MainWindow::openFile);
    connect(ui->actionSave, &QAction::triggered, this, &MainWindow::saveFile);
    connect(ui->actionRotate, &QAction::triggered, this, &MainWindow::rotateImage);
    connect(ui->actionZoomIn, &QAction::triggered, this, &MainWindow::zoomInImage);
    connect(ui->actionZoomOut, &QAction::triggered, this, &MainWindow::zoomOutImage);

    // Connect ImageProcessor's signal to displayImage slot
    connect(imageProcessor, &ImageProcessor::imageProcessed, this, &MainWindow::displayImage);
}

MainWindow::~MainWindow()
{
    delete ui;
    delete imageProcessor;
}

void MainWindow::openFile()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Image"), "", tr("Image Files (*.png *.jpg *.bmp *.jpeg);;All Files (*)"));
    if (!fileName.isEmpty()) {
        cv::Mat loadedImage;
        if (imageProcessor->openImage(fileName.toStdString(), loadedImage)) {
            currentImage = loadedImage.clone(); // Clone loaded image
            displayImage(currentImage);
        }
        else {
            QMessageBox::critical(this, tr("Error"), tr("Failed to open image file"));
        }
    }
}

void MainWindow::saveFile()
{
    if (!currentImage.empty()) {
        QString filePath = QFileDialog::getSaveFileName(this, tr("Save Image"), "", tr("Images (*.png *.jpg *.bmp)"));
        if (!filePath.isEmpty()) {
            if (!imageProcessor->saveImage(filePath.toStdString(), currentImage)) {
                QMessageBox::critical(this, tr("Error"), tr("Failed to save image"));
            }
        }
    }
    else {
        QMessageBox::critical(this, tr("Error"), tr("No image to save"));
    }
}

void MainWindow::rotateImage()
{
    if (!currentImage.empty()) {
        auto future = imageProcessor->rotateImage(currentImage);
        future.waitForFinished();
        if (future.result()) {
            displayImage(currentImage);
        }
        else {
            qDebug() << "Failed to rotate image.";
        }
    }
}

void MainWindow::zoomInImage()
{
    if (!currentImage.empty()) {
        scaleFactor = 1.25;
        auto future = imageProcessor->zoomImage(currentImage,
            scaleFactor);
        future.waitForFinished();
        if (future.result()) {
            displayImage(currentImage);
        }else {
            qDebug() << "Failed to zoom in image.";
        }
    }
}

void MainWindow::zoomOutImage()
{
    if (!currentImage.empty()) {
        scaleFactor = 0.8;
        auto future = imageProcessor->zoomImage(currentImage,
            scaleFactor);
        future.waitForFinished();
        if (future.result()) {
            displayImage(currentImage);
        }else {
            qDebug() << "Failed to zoom out image.";
        }
    }
}

void MainWindow::displayImage(const cv::Mat& image)
{
    QImage qImage(image.data,
        image.cols,
        image.rows,
        static_cast<int>(image.step),
        QImage::Format_BGR888);
    ui->label->setPixmap(QPixmap::fromImage(qImage));
    ui->label->adjustSize();
}
