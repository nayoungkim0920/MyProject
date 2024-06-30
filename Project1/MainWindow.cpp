//MainWindow.cpp
#include "MainWindow.h"

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , imageProcessor(new ImageProcessor)
    , scaleFactor(1.0)
{
    ui->setupUi(this);

    connectActions();

    connectImageProcessor();

    setInitialWindowGeometry();
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
            initialImage = currentImage.clone();
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

        if (!future.result()) {
            qDebug() << "Failed to apply rotateImage.";
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

        if (!future.result()) {
            qDebug() << "Failed to apply zoomInImage.";
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

        if (!future.result()) {
            qDebug() << "Failed to apply zoomOutImage.";
        }
    }
}

void MainWindow::convertToGrayscale()
{
    if (!currentImage.empty()) {
        qDebug() << "convertToGrayscale() currentImage type : " << currentImage.type();
        qDebug() << "convertToGrayscale() currentImage channels : " << currentImage.channels();

        auto future = imageProcessor->convertToGrayscaleAsync(currentImage);
        future.waitForFinished();

        if (!future.result()) {
            qDebug() << "Failed to apply convertToGrayscale.";
        }
    }
}

void MainWindow::applyGaussianBlur()
{
    bool ok;
    int kernelSize = QInputDialog::getInt(this,
        tr("Gaussian Blur"), 
        tr("Enter kernel size (odd nubmber):"),
        5, 1, 101, 2, &ok);

    if (ok) {
        applyImageProcessing(&ImageProcessor::applyGaussianBlur, currentImage, kernelSize);
    }
}

void MainWindow::cannyEdges()
{
    applyImageProcessing(&ImageProcessor::cannyEdges, currentImage);
}

void MainWindow::medianFilter()
{
    applyImageProcessing(&ImageProcessor::medianFilter, currentImage);
}

void MainWindow::laplacianFilter()
{
    applyImageProcessing(&ImageProcessor::laplacianFilter, currentImage);
}

void MainWindow::bilateralFilter()
{
    applyImageProcessing(&ImageProcessor::bilateralFilter, currentImage);
}

void MainWindow::exitApplication()
{
    QApplication::quit();
}

void MainWindow::redoAction()
{
    if (imageProcessor->canRedo()) {
        imageProcessor->redo();
    }
}

void MainWindow::undoAction()
{
    if (imageProcessor->canUndo()) {
        imageProcessor->undo();
    }
}

void MainWindow::first()
{
    //초기 이미지로 되돌리기
    if (!initialImage.empty()) {
        currentImage = initialImage.clone();
        displayImage(currentImage);
        imageProcessor->cleanUndoStack();
        imageProcessor->cleanRedoStack();
    }
    else {
        QMessageBox::warning(this,
            tr("Warning"),
            tr("No initial Image available."));
        return;
    }
}

void MainWindow::displayImage(const cv::Mat& image)
{
    QMetaObject::invokeMethod(this, [this, image]() {

        qDebug() << "displayImage() channels : " << image.channels();

        currentImage = image;

        // 이미지 타입이 그레이스케일(CV_8UC1)인지 확인합니다.
        if (image.type() == CV_8UC1) {
            qDebug() << "displayImage() type : graysclae CV_8UC1 Format_Grayscale8";
            QImage qImage(image.data,
                image.cols,
                image.rows,
                static_cast<int>(image.step),
                QImage::Format_Grayscale8);
            ui->label->setPixmap(QPixmap::fromImage(qImage));
            ui->label->adjustSize();
        }
        else {
            qDebug() << "displayImage() type : Format_BGR888";
            QImage qImage(image.data,
                image.cols,
                image.rows,
                static_cast<int>(image.step),
                QImage::Format_BGR888);
            ui->label->setPixmap(QPixmap::fromImage(qImage));
            ui->label->adjustSize();
        }

        });  

}


void MainWindow::connectActions()
{
    connect(ui->actionOpen, &QAction::triggered, this, &MainWindow::openFile);
    connect(ui->actionSave, &QAction::triggered, this, &MainWindow::saveFile);
    connect(ui->actionExit, &QAction::triggered, this, &MainWindow::exitApplication);

    connect(ui->actionRotate, &QAction::triggered, this, &MainWindow::rotateImage);
    connect(ui->actionZoomIn, &QAction::triggered, this, &MainWindow::zoomInImage);
    connect(ui->actionZoomOut, &QAction::triggered, this, &MainWindow::zoomOutImage);
    connect(ui->actionRedo, &QAction::triggered, this, &MainWindow::redoAction);
    connect(ui->actionUndo, &QAction::triggered, this, &MainWindow::undoAction);

    connect(ui->actionGrayscale, &QAction::triggered, this, &MainWindow::convertToGrayscale);
    connect(ui->actionGaussianBlur, &QAction::triggered, this, &MainWindow::applyGaussianBlur);
    connect(ui->actionCannyEdges, &QAction::triggered, this, &MainWindow::cannyEdges);
    connect(ui->actionMedianFilter, &QAction::triggered, this, &MainWindow::medianFilter);
    connect(ui->actionLaplacianFilter, &QAction::triggered, this, &MainWindow::laplacianFilter);
    connect(ui->actionBilateralFilter, &QAction::triggered, this, &MainWindow::bilateralFilter);

    connect(ui->actionFirst, &QAction::triggered, this, &MainWindow::first);
    
}

void MainWindow::connectImageProcessor()
{
    // Connect ImageProcessor's signal to displayImage slot
    connect(imageProcessor, &ImageProcessor::imageProcessed, this, &MainWindow::displayImage);
}

void MainWindow::setInitialWindowGeometry()
{
    const int initialWidth = 800;
    const int initialHeight = 600;
    const int initialX = 100;
    const int initialY = 100;
    this->setGeometry(initialX, initialY, initialWidth, initialHeight);
}

template<typename Func, typename ...Args>
inline void MainWindow::applyImageProcessing(Func func, Args&& ...args)
{
    if (!currentImage.empty()) {
        auto future = (imageProcessor->*func)(std::forward<Args>(args)...);
        future.waitForFinished();
        if (!future.result()) {
            qDebug() << "Failed to apply" << Q_FUNC_INFO;
        }
    }
    else {
        qDebug() << "No image to process.";
    }
}