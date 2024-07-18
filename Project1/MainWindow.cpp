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
    ui->label_npp_title->setText(QString("NPP"));
    ui->label_cuda_title->setText(QString("CUDA"));
    ui->label_cudakernel_title->setText(QString("CUDA Kernel"));
    ui->label_gstreamer_title->setText(QString("GStreamer"));
    

    connectActions();

    //ó���ε� �� ����ó���� �ʹ� ���� �߰���
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

            //�̹���ũ�⸦ 400*300 ����
            cv::resize(loadedImage, loadedImage, cv::Size(400, 300));

            currentImageOpenCV = loadedImage.clone();
            currentImageIPP = loadedImage.clone();
            currentImageCUDA = loadedImage.clone();
            currentImageCUDAKernel = loadedImage.clone();
            currentImageNPP = loadedImage.clone();
            currentImageGStreamer = loadedImage.clone();

            initialImageOpenCV = currentImageOpenCV.clone();
            initialImageIPP = currentImageIPP.clone();
            initialImageCUDA = currentImageCUDA.clone();
            initialImageCUDAKernel = currentImageCUDAKernel.clone();
            initialImageNPP = currentImageNPP.clone();
            initialImageGStreamer = currentImageGStreamer.clone();

            displayImage(initialImageOpenCV, ui->label_opencv);
            displayImage(initialImageIPP, ui->label_ipp);
            displayImage(initialImageCUDA, ui->label_cuda);
            displayImage(initialImageCUDAKernel, ui->label_cudakernel);
            displayImage(initialImageNPP, ui->label_npp);
            displayImage(initialImageGStreamer, ui->label_gstreamer);
        }
        else {
            QMessageBox::critical(this, tr("Error"), tr("Failed to open image file"));
        }
    }
}

void MainWindow::saveFile()
{
    //if (!currentImage.empty()) {
    //    QString filePath = QFileDialog::getSaveFileName(this, tr("Save Image"), "", tr("Images (*.png *.jpg *.bmp)"));
    //    if (!filePath.isEmpty()) {
    //        if (!imageProcessor->saveImage(filePath.toStdString(), currentImage)) {
    //            QMessageBox::critical(this, tr("Error"), tr("Failed to save image"));
    //        }
    //    }
    //}
    //else {
    //    QMessageBox::critical(this, tr("Error"), tr("No image to save"));
    //}
}

void MainWindow::rotateImage()
{
    QtConcurrent::run([this]() {
        if (!currentImageOpenCV.empty()) {
            imageProcessor->rotateImage(currentImageOpenCV
                , currentImageIPP
                , currentImageCUDA
                , currentImageCUDAKernel
                , currentImageNPP);
        }
        });
    ////applyImageProcessing(&ImageProcessor::rotateImage, currentImage);
}

void MainWindow::zoomInImage()
{
    QtConcurrent::run([this]() {
        if (!currentImageOpenCV.empty()) {
            imageProcessor->zoomInImage(currentImageOpenCV
                , currentImageIPP
                , currentImageCUDA
                , currentImageCUDAKernel
                , scaleFactor = 1.25);
        }
    });
    //applyImageProcessing(&ImageProcessor::zoominImage, currentImage, scaleFactor=1.25);
}

void MainWindow::zoomOutImage()
{
    QtConcurrent::run([this]() {
        if (!currentImageOpenCV.empty()) {
            imageProcessor->zoomOutImage(currentImageOpenCV
                , currentImageIPP
                , currentImageCUDA
                , currentImageCUDAKernel
                , scaleFactor = 0.8);
        }
        });
    //applyImageProcessing(&ImageProcessor::zoomoutImage, currentImage, scaleFactor = 0.8);
}

void MainWindow::grayScale()
{
    QtConcurrent::run([this]() {
        
        imageProcessor->grayScale(currentImageOpenCV
        , currentImageIPP
        , currentImageCUDA
        , currentImageCUDAKernel);

        });

    //applyImageProcessing(&ImageProcessor::grayScale, currentImage);
}

void MainWindow::gaussianBlur()
{
    bool ok;

    QInputDialog inputDialog(this);
    inputDialog.setWindowTitle(tr("Gaussian Blur"));
    inputDialog.setLabelText(tr("Enter kernel size (odd number):"));
    inputDialog.setIntRange(1, 101);
    inputDialog.setIntStep(2);
    inputDialog.setIntValue(5);

    // �ּ� ũ�� ����
    inputDialog.setMinimumSize(200, 100);
    inputDialog.resize(200, 100);

    // ���� �������� ��ġ�� ũ�⸦ ����
    QRect windowGeometry = geometry();
    int x = windowGeometry.x() + (windowGeometry.width() - inputDialog.width()) / 2;
    int y = windowGeometry.y() + (windowGeometry.height() - inputDialog.height()) / 2;

    // ��ġ ����
    inputDialog.move(x, y);

    if (inputDialog.exec() == QDialog::Accepted) {
        int kernelSize = inputDialog.intValue();
        QtConcurrent::run([this, kernelSize]() {
            imageProcessor->gaussianBlur(currentImageOpenCV, currentImageIPP, currentImageCUDA, currentImageCUDAKernel, kernelSize);
            });
    }
}



void MainWindow::cannyEdges()
{
    QtConcurrent::run([this]() {
            imageProcessor->cannyEdges(currentImageOpenCV, currentImageIPP, currentImageCUDA, currentImageCUDAKernel);
        });
    //applyImageProcessing(&ImageProcessor::cannyEdges, currentImage);
}

void MainWindow::medianFilter()
{
    QtConcurrent::run([this]() {
            imageProcessor->medianFilter(currentImageOpenCV, currentImageIPP, currentImageCUDA, currentImageCUDAKernel);
        });
    //applyImageProcessing(&ImageProcessor::medianFilter, currentImage);
}

void MainWindow::laplacianFilter()
{
    QtConcurrent::run([this]() {
        imageProcessor->laplacianFilter(currentImageOpenCV, currentImageIPP, currentImageCUDA, currentImageCUDAKernel);
        });
    //applyImageProcessing(&ImageProcessor::laplacianFilter, currentImage);
}

void MainWindow::bilateralFilter()
{
    QtConcurrent::run([this]() {
            imageProcessor->bilateralFilter(currentImageOpenCV, currentImageIPP, currentImageCUDA, currentImageCUDAKernel, currentImageNPP, currentImageGStreamer);
        });
    //applyImageProcessing(&ImageProcessor::bilateralFilter, currentImage);
}

void MainWindow::sobelFilter()
{
    QtConcurrent::run([this]() {
            imageProcessor->sobelFilter(currentImageOpenCV, currentImageIPP, currentImageCUDA, currentImageCUDAKernel);
        });
    //applyImageProcessing(&ImageProcessor::)
}

void MainWindow::exitApplication()
{
    QApplication::quit();
}

void MainWindow::redoAction()
{
    if (imageProcessor->canRedoOpenCV()) {
        imageProcessor->redo();
    }
}

void MainWindow::undoAction()
{
    if (imageProcessor->canUndoOpenCV()) {
        imageProcessor->undo();
    }
}

void MainWindow::first()
{
    //�ʱ� �̹����� �ǵ�����
    //if (!initialImage.empty()) {
    //    currentImage = initialImage.clone();

    //    displayImage(currentImage, ui->label_opencv);
    //    displayImage(currentImage, ui->label_ipp);
     //   displayImage(currentImage, ui->label_cuda);
     //   displayImage(currentImage, ui->label_cudakernel);

    //    imageProcessor->cleanUndoStack();
    //    imageProcessor->cleanRedoStack();
    //}
    //else {
    //    QMessageBox::warning(this,
   //         tr("Warning"),
   //         tr("No initial Image available."));
   //     return;
   // }
}

void MainWindow::displayImage(cv::Mat image, QLabel* label)
{
    // �̹��� Ÿ�Կ� ���� QImage�� �����մϴ�.
    QImage qImage;

    qDebug() << "displayImage() called with image type:" << image.type();

    // OpenCV�� Mat �̹��� Ÿ�Կ� ���� �ٸ� QImage ������ ����մϴ�.
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
            QImage::Format_RGB888).rgbSwapped(); // BGR -> RGB ������ ��ȯ
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

        // 16-bit �̹����� 8-bit�� ��ȯ
        cv::Mat temp;
        image.convertTo(temp, CV_8UC3, 1.0 / 256.0);
        qImage = QImage(temp.data,
            temp.cols,
            temp.rows,
            static_cast<int>(temp.step),
            QImage::Format_RGB888).rgbSwapped(); // BGR -> RGB ������ ��ȯ
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
        return; // �������� �ʴ� �̹��� Ÿ���� ó������ ����
    }

    // QLabel ������ QPixmap���� �̹����� �����մϴ�.
    QPixmap pixmap = QPixmap::fromImage(qImage);
    label->setPixmap(pixmap);
    label->setScaledContents(false); // �̹����� Label ũ�⿡ �°� ����
    label->adjustSize(); // Label ũ�� ����
    qDebug() << "displayImage() finished";
}


void MainWindow::handleImageProcessed(QVector<ImageProcessor::ProcessingResult> results)
{
    for (int i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        if (i == 0) {
            currentImageOpenCV = result.processedImage.clone();
            displayImage(result.processedImage, ui->label_opencv);
            ui->label_opencv_title->setText(QString("%1 %2 %3ms %4 %5 %6")
                .arg(result.processName)
                .arg(result.functionName)
                .arg(result.processingTime)
                .arg(result.argInfo)
                .arg(result.inputInfo)
                .arg(result.outputInfo));
        }
        else if (i == 1) {
            currentImageIPP = result.processedImage;
            displayImage(result.processedImage, ui->label_ipp);
            ui->label_ipp_title->setText(QString("%1 %2 %3ms %4 %5 %6")
                .arg(result.processName)
                .arg(result.functionName)
                .arg(result.processingTime)
                .arg(result.argInfo)
                .arg(result.inputInfo)
                .arg(result.outputInfo));
        }
        else if (i == 2) {
            currentImageCUDA= result.processedImage;
            displayImage(result.processedImage, ui->label_cuda);
            ui->label_cuda_title->setText(QString("%1 %2 %3ms %4 %5 %6")
                .arg(result.processName)
                .arg(result.functionName)
                .arg(result.processingTime)
                .arg(result.argInfo)
                .arg(result.inputInfo)
                .arg(result.outputInfo));
        }
        else if (i == 3) {
            currentImageCUDAKernel = result.processedImage;
            displayImage(result.processedImage, ui->label_cudakernel);
            ui->label_cudakernel_title->setText(QString("%1 %2 %3ms %4 %5 %6")
                .arg(result.processName)
                .arg(result.functionName)
                .arg(result.processingTime)
                .arg(result.argInfo)
                .arg(result.inputInfo)
                .arg(result.outputInfo));
        }
        else if (i == 4) {
            currentImageNPP = result.processedImage;
            displayImage(result.processedImage, ui->label_npp);
            ui->label_npp_title->setText(QString("%1 %2 %3ms %4 %5 %6")
                .arg(result.processName)
                .arg(result.functionName)
                .arg(result.processingTime)
                .arg(result.argInfo)
                .arg(result.inputInfo)
                .arg(result.outputInfo));
        }
        else if (i == 5) {
            currentImageGStreamer = result.processedImage;
            displayImage(result.processedImage, ui->label_gstreamer);
            ui->label_gstreamer_title->setText(QString("%1 %2 %3ms %4 %5 %6")
                .arg(result.processName)
                .arg(result.functionName)
                .arg(result.processingTime)
                .arg(result.argInfo)
                .arg(result.inputInfo)
                .arg(result.outputInfo));
        }

            
    }

    // �̹��� ���
    //displayImage(processedImage);

    // ���� ǥ���ٿ� ó�� �ð� ���
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