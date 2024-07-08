//MainWindow.cpp
#include "MainWindow.h"

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , imageProcessor(new ImageProcessor)
    , scaleFactor(1.0)
{
    ui->setupUi(this);

    ui->label_opencv_title->setText(QString("OpenCV"));
    ui->label_ipp_title->setText(QString("IPP"));
    ui->label_cuda_title->setText(QString("CUDA"));
    ui->label_cudakernel_title->setText(QString("CUDA Kernel"));

    connectActions();

    //처음로딩 후 필터처리가 너무 느려 추가함
    imageProcessor->initializeCUDA();

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

            displayImage(currentImage, ui->label_opencv);
            displayImage(currentImage, ui->label_ipp);
            displayImage(currentImage, ui->label_cuda);
            displayImage(currentImage, ui->label_cudakernel);
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
    QtConcurrent::run([this]() {
        if (!currentImage.empty()) {
            imageProcessor->rotateImage(currentImage);
        }
    });
    //applyImageProcessing(&ImageProcessor::rotateImage, currentImage);
}

void MainWindow::zoomInImage()
{
    QtConcurrent::run([this]() {
        if (!currentImage.empty()) {
            imageProcessor->zoominImage(currentImage, scaleFactor = 1.25);
        }
    });
    //applyImageProcessing(&ImageProcessor::zoominImage, currentImage, scaleFactor=1.25);
}

void MainWindow::zoomOutImage()
{
    QtConcurrent::run([this]() {
        if (!currentImage.empty()) {
            imageProcessor->zoomoutImage(currentImage, scaleFactor = 0.8);
        }
        });
    //applyImageProcessing(&ImageProcessor::zoomoutImage, currentImage, scaleFactor = 0.8);
}

void MainWindow::grayScale()
{
    QtConcurrent::run([this]() {
        if (!currentImage.empty()) {
            imageProcessor->grayScale(currentImage);
        }
        });
    //applyImageProcessing(&ImageProcessor::grayScale, currentImage);
}

void MainWindow::gaussianBlur()
{
    bool ok;
    int kernelSize = QInputDialog::getInt(this,
        tr("Gaussian Blur"),
        tr("Enter kernel size (odd nubmber):"),
        5, 1, 101, 2, &ok);

    if (ok) {
        QtConcurrent::run([this, kernelSize]() {
            if (!currentImage.empty()) {
                imageProcessor->gaussianBlur(currentImage, kernelSize);
            }
            });
        //applyImageProcessing(&ImageProcessor::gaussianBlur, currentImage, kernelSize);
    }
}

void MainWindow::cannyEdges()
{
    QtConcurrent::run([this]() {
        if (!currentImage.empty()) {
            imageProcessor->cannyEdges(currentImage);
        }
        });
    //applyImageProcessing(&ImageProcessor::cannyEdges, currentImage);
}

void MainWindow::medianFilter()
{
    QtConcurrent::run([this]() {
        if (!currentImage.empty()) {
            imageProcessor->medianFilter(currentImage);
        }
        });
    //applyImageProcessing(&ImageProcessor::medianFilter, currentImage);
}

void MainWindow::laplacianFilter()
{
    QtConcurrent::run([this]() {
        if (!currentImage.empty()) {
            imageProcessor->laplacianFilter(currentImage);
        }
        });
    //applyImageProcessing(&ImageProcessor::laplacianFilter, currentImage);
}

void MainWindow::bilateralFilter()
{
    QtConcurrent::run([this]() {
        if (!currentImage.empty()) {
            imageProcessor->bilateralFilter(currentImage);
        }
        });
    //applyImageProcessing(&ImageProcessor::bilateralFilter, currentImage);
}

void MainWindow::sobelFilter()
{
    QtConcurrent::run([this]() {
        if (!currentImage.empty()) {
            imageProcessor->sobelFilter(currentImage);
        }
        });
    //applyImageProcessing(&ImageProcessor::)
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

        displayImage(currentImage, ui->label_opencv);
        displayImage(currentImage, ui->label_ipp);
        displayImage(currentImage, ui->label_cuda);
        displayImage(currentImage, ui->label_cudakernel);

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

void MainWindow::displayImage(cv::Mat image, QLabel* label)
{
    QMetaObject::invokeMethod(this, [this, image, label]() {
        qDebug() << "displayImage() channels: " << image.channels();

        currentImage = image;

        // 이미지 타입에 따라 QImage를 생성합니다.
        QImage qImage;
        if (image.type() == CV_8UC1) {
            qDebug() << "displayImage() type: grayscale CV_8UC1 Format_Grayscale8";
            qImage = QImage(image.data,
                image.cols,
                image.rows,
                static_cast<int>(image.step),
                QImage::Format_Grayscale8);
        }
        else if (image.type() == CV_8UC3) {
            qDebug() << "displayImage() type: BGR CV_8UC3 Format_RGB888";
            qImage = QImage(image.data,
                image.cols,
                image.rows,
                static_cast<int>(image.step),
                QImage::Format_RGB888).rgbSwapped();
        }
        else {
            qDebug() << "displayImage() type: not supported";
            return; // 지원하지 않는 이미지 타입은 처리하지 않음
        }

        // QLabel 위젯에 QPixmap으로 이미지를 설정합니다.
        QPixmap pixmap = QPixmap::fromImage(qImage);
        label->setPixmap(pixmap);
        });
}

void MainWindow::handleImageProcessed(QVector<ImageProcessor::ProcessingResult> results)
{
    for (int i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        if (i == 0) {
            displayImage(result.processedImage, ui->label_opencv);
            ui->label_opencv_title->setText(QString("%1 %2 %3ms")
                .arg(result.processName)
                .arg(result.functionName)
                .arg(result.processingTime));
        }
        else if (i == 1) {
            displayImage(result.processedImage, ui->label_ipp);
            ui->label_ipp_title->setText(QString("%1 %2 %3ms")
                .arg(result.processName)
                .arg(result.functionName)
                .arg(result.processingTime));
        }
        else if (i == 2) {
            displayImage(result.processedImage, ui->label_cuda);
            ui->label_cuda_title->setText(QString("%1 %2 %3ms")
                .arg(result.processName)
                .arg(result.functionName)
                .arg(result.processingTime));
        }
        else if (i == 3) {
            displayImage(result.processedImage, ui->label_cudakernel);
            ui->label_cudakernel_title->setText(QString("%1 %2 %3ms")
                .arg(result.processName)
                .arg(result.functionName)
                .arg(result.processingTime));
        }
            
    }

    // 이미지 출력
    //displayImage(processedImage);

    // 상태 표시줄에 처리 시간 출력
    //statusBar()->showMessage(
    //    QString("%1 processed in %2 ms").arg(processName).arg(processingTimeMs));

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

    connect(ui->actionGrayscale, &QAction::triggered, this, &MainWindow::grayScale);
    connect(ui->actionGaussianBlur, &QAction::triggered, this, &MainWindow::gaussianBlur);
    connect(ui->actionCannyEdges, &QAction::triggered, this, &MainWindow::cannyEdges);
    connect(ui->actionMedianFilter, &QAction::triggered, this, &MainWindow::medianFilter);
    connect(ui->actionLaplacianFilter, &QAction::triggered, this, &MainWindow::laplacianFilter);
    connect(ui->actionBilateralFilter, &QAction::triggered, this, &MainWindow::bilateralFilter);
    connect(ui->actionSobelFilter, &QAction::triggered, this, &MainWindow::sobelFilter);

    connect(ui->actionFirst, &QAction::triggered, this, &MainWindow::first);

}

void MainWindow::connectImageProcessor()
{
    // Connect ImageProcessor's signal to displayImage slot
    connect(imageProcessor, &ImageProcessor::imageProcessed, this, &MainWindow::handleImageProcessed);
}

void MainWindow::setInitialWindowGeometry()
{
    const int initialWidth = 800;
    const int initialHeight = 600;
    const int initialX = 100;
    const int initialY = 100;
    this->setGeometry(initialX, initialY, initialWidth, initialHeight);
}

//template<typename Func, typename ...Args>
//inline void MainWindow::applyImageProcessing(Func func, Args&& ...args)
//{
//    if (!currentImage.empty()) {
//        auto future = (imageProcessor->*func)(std::forward<Args>(args)...);
//        future.waitForFinished();
//        if (!future.result()) {
//            qDebug() << "Failed to apply" << Q_FUNC_INFO;
//        }
//    }
//    else {
//        qDebug() << "No image to process.";
//    }
//}