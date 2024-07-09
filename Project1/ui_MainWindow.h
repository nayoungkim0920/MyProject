/********************************************************************************
** Form generated from reading UI file 'MainWindow.ui'
**
** Created by: Qt User Interface Compiler version 6.7.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionOpen;
    QAction *actionSave;
    QAction *actionExit;
    QAction *actionRotate;
    QAction *actionZoomIn;
    QAction *actionZoomOut;
    QAction *actionUndo;
    QAction *actionRedo;
    QAction *actionFirst;
    QAction *actionGrayscale;
    QAction *actionGaussianBlur;
    QAction *actionCannyEdges;
    QAction *actionMedianFilter;
    QAction *actionSobelFilter;
    QAction *actionLaplacianFilter;
    QAction *actionBilateralFilter;
    QWidget *centralwidget;
    QVBoxLayout *verticalLayout;
    QScrollArea *scrollArea;
    QWidget *scrollAreaWidgetContents;
    QVBoxLayout *verticalLayout_scroll;
    QVBoxLayout *verticalLayout_images_and_times;
    QHBoxLayout *horizontalLayout_row1;
    QLabel *label_opencv;
    QLabel *label_ipp;
    QHBoxLayout *horizontalLayout_row2;
    QLabel *label_opencv_title;
    QLabel *label_ipp_title;
    QHBoxLayout *horizontalLayout_row3;
    QLabel *label_cuda;
    QLabel *label_cudakernel;
    QHBoxLayout *horizontalLayout_row4;
    QLabel *label_cuda_title;
    QLabel *label_cudakernel_title;
    QMenuBar *menubar;
    QMenu *menuFile;
    QMenu *menuEdit;
    QMenu *menuFilters;
    QStatusBar *statusbar;
    QToolBar *fileToolBar;
    QToolBar *filtersToolBar;
    QToolBar *mainToolBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName("MainWindow");
        MainWindow->resize(800, 600);
        actionOpen = new QAction(MainWindow);
        actionOpen->setObjectName("actionOpen");
        actionSave = new QAction(MainWindow);
        actionSave->setObjectName("actionSave");
        actionExit = new QAction(MainWindow);
        actionExit->setObjectName("actionExit");
        actionRotate = new QAction(MainWindow);
        actionRotate->setObjectName("actionRotate");
        actionZoomIn = new QAction(MainWindow);
        actionZoomIn->setObjectName("actionZoomIn");
        actionZoomOut = new QAction(MainWindow);
        actionZoomOut->setObjectName("actionZoomOut");
        actionUndo = new QAction(MainWindow);
        actionUndo->setObjectName("actionUndo");
        actionRedo = new QAction(MainWindow);
        actionRedo->setObjectName("actionRedo");
        actionFirst = new QAction(MainWindow);
        actionFirst->setObjectName("actionFirst");
        actionGrayscale = new QAction(MainWindow);
        actionGrayscale->setObjectName("actionGrayscale");
        actionGaussianBlur = new QAction(MainWindow);
        actionGaussianBlur->setObjectName("actionGaussianBlur");
        actionCannyEdges = new QAction(MainWindow);
        actionCannyEdges->setObjectName("actionCannyEdges");
        actionMedianFilter = new QAction(MainWindow);
        actionMedianFilter->setObjectName("actionMedianFilter");
        actionSobelFilter = new QAction(MainWindow);
        actionSobelFilter->setObjectName("actionSobelFilter");
        actionLaplacianFilter = new QAction(MainWindow);
        actionLaplacianFilter->setObjectName("actionLaplacianFilter");
        actionBilateralFilter = new QAction(MainWindow);
        actionBilateralFilter->setObjectName("actionBilateralFilter");
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName("centralwidget");
        verticalLayout = new QVBoxLayout(centralwidget);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName("verticalLayout");
        scrollArea = new QScrollArea(centralwidget);
        scrollArea->setObjectName("scrollArea");
        scrollArea->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName("scrollAreaWidgetContents");
        verticalLayout_scroll = new QVBoxLayout(scrollAreaWidgetContents);
        verticalLayout_scroll->setSpacing(6);
        verticalLayout_scroll->setContentsMargins(11, 11, 11, 11);
        verticalLayout_scroll->setObjectName("verticalLayout_scroll");
        verticalLayout_images_and_times = new QVBoxLayout();
        verticalLayout_images_and_times->setSpacing(6);
        verticalLayout_images_and_times->setObjectName("verticalLayout_images_and_times");
        horizontalLayout_row1 = new QHBoxLayout();
        horizontalLayout_row1->setSpacing(6);
        horizontalLayout_row1->setObjectName("horizontalLayout_row1");
        label_opencv = new QLabel(scrollAreaWidgetContents);
        label_opencv->setObjectName("label_opencv");
        label_opencv->setAlignment(Qt::AlignCenter);
        label_opencv->setScaledContents(true);

        horizontalLayout_row1->addWidget(label_opencv);

        label_ipp = new QLabel(scrollAreaWidgetContents);
        label_ipp->setObjectName("label_ipp");
        label_ipp->setAlignment(Qt::AlignCenter);
        label_ipp->setScaledContents(true);

        horizontalLayout_row1->addWidget(label_ipp);


        verticalLayout_images_and_times->addLayout(horizontalLayout_row1);

        horizontalLayout_row2 = new QHBoxLayout();
        horizontalLayout_row2->setSpacing(6);
        horizontalLayout_row2->setObjectName("horizontalLayout_row2");
        label_opencv_title = new QLabel(scrollAreaWidgetContents);
        label_opencv_title->setObjectName("label_opencv_title");
        label_opencv_title->setAlignment(Qt::AlignLeft);

        horizontalLayout_row2->addWidget(label_opencv_title);

        label_ipp_title = new QLabel(scrollAreaWidgetContents);
        label_ipp_title->setObjectName("label_ipp_title");
        label_ipp_title->setAlignment(Qt::AlignLeft);

        horizontalLayout_row2->addWidget(label_ipp_title);


        verticalLayout_images_and_times->addLayout(horizontalLayout_row2);

        horizontalLayout_row3 = new QHBoxLayout();
        horizontalLayout_row3->setSpacing(6);
        horizontalLayout_row3->setObjectName("horizontalLayout_row3");
        label_cuda = new QLabel(scrollAreaWidgetContents);
        label_cuda->setObjectName("label_cuda");
        label_cuda->setAlignment(Qt::AlignCenter);
        label_cuda->setScaledContents(true);

        horizontalLayout_row3->addWidget(label_cuda);

        label_cudakernel = new QLabel(scrollAreaWidgetContents);
        label_cudakernel->setObjectName("label_cudakernel");
        label_cudakernel->setAlignment(Qt::AlignCenter);
        label_cudakernel->setScaledContents(true);

        horizontalLayout_row3->addWidget(label_cudakernel);


        verticalLayout_images_and_times->addLayout(horizontalLayout_row3);

        horizontalLayout_row4 = new QHBoxLayout();
        horizontalLayout_row4->setSpacing(6);
        horizontalLayout_row4->setObjectName("horizontalLayout_row4");
        label_cuda_title = new QLabel(scrollAreaWidgetContents);
        label_cuda_title->setObjectName("label_cuda_title");
        label_cuda_title->setAlignment(Qt::AlignLeft);

        horizontalLayout_row4->addWidget(label_cuda_title);

        label_cudakernel_title = new QLabel(scrollAreaWidgetContents);
        label_cudakernel_title->setObjectName("label_cudakernel_title");
        label_cudakernel_title->setAlignment(Qt::AlignLeft);

        horizontalLayout_row4->addWidget(label_cudakernel_title);


        verticalLayout_images_and_times->addLayout(horizontalLayout_row4);


        verticalLayout_scroll->addLayout(verticalLayout_images_and_times);

        scrollArea->setWidget(scrollAreaWidgetContents);

        verticalLayout->addWidget(scrollArea);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName("menubar");
        menubar->setGeometry(QRect(0, 0, 800, 26));
        menuFile = new QMenu(menubar);
        menuFile->setObjectName("menuFile");
        menuEdit = new QMenu(menubar);
        menuEdit->setObjectName("menuEdit");
        menuFilters = new QMenu(menubar);
        menuFilters->setObjectName("menuFilters");
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName("statusbar");
        MainWindow->setStatusBar(statusbar);
        fileToolBar = new QToolBar(MainWindow);
        fileToolBar->setObjectName("fileToolBar");
        MainWindow->addToolBar(Qt::ToolBarArea::TopToolBarArea, fileToolBar);
        filtersToolBar = new QToolBar(MainWindow);
        filtersToolBar->setObjectName("filtersToolBar");
        MainWindow->addToolBar(Qt::ToolBarArea::TopToolBarArea, filtersToolBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName("mainToolBar");
        MainWindow->addToolBar(Qt::ToolBarArea::TopToolBarArea, mainToolBar);

        menubar->addAction(menuFile->menuAction());
        menubar->addAction(menuEdit->menuAction());
        menubar->addAction(menuFilters->menuAction());
        menuFile->addAction(actionOpen);
        menuFile->addAction(actionSave);
        menuFile->addAction(actionExit);
        menuEdit->addAction(actionRotate);
        menuEdit->addAction(actionZoomIn);
        menuEdit->addAction(actionZoomOut);
        menuEdit->addAction(actionUndo);
        menuEdit->addAction(actionRedo);
        menuEdit->addAction(actionFirst);
        menuFilters->addAction(actionGrayscale);
        menuFilters->addAction(actionGaussianBlur);
        menuFilters->addAction(actionCannyEdges);
        menuFilters->addAction(actionMedianFilter);
        menuFilters->addAction(actionSobelFilter);
        menuFilters->addAction(actionLaplacianFilter);
        menuFilters->addAction(actionBilateralFilter);
        fileToolBar->addAction(actionOpen);
        fileToolBar->addAction(actionSave);
        fileToolBar->addAction(actionExit);
        filtersToolBar->addAction(actionGrayscale);
        filtersToolBar->addAction(actionGaussianBlur);
        filtersToolBar->addAction(actionCannyEdges);
        filtersToolBar->addAction(actionMedianFilter);
        filtersToolBar->addAction(actionSobelFilter);
        filtersToolBar->addAction(actionLaplacianFilter);
        filtersToolBar->addAction(actionBilateralFilter);
        mainToolBar->addAction(actionRotate);
        mainToolBar->addAction(actionZoomIn);
        mainToolBar->addAction(actionZoomOut);
        mainToolBar->addAction(actionUndo);
        mainToolBar->addAction(actionRedo);
        mainToolBar->addAction(actionFirst);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        actionOpen->setText(QCoreApplication::translate("MainWindow", "&Open", nullptr));
        actionSave->setText(QCoreApplication::translate("MainWindow", "&Save", nullptr));
        actionExit->setText(QCoreApplication::translate("MainWindow", "E&xit", nullptr));
        actionRotate->setText(QCoreApplication::translate("MainWindow", "&Rotate", nullptr));
        actionZoomIn->setText(QCoreApplication::translate("MainWindow", "Zoom &In", nullptr));
        actionZoomOut->setText(QCoreApplication::translate("MainWindow", "Zoom &Out", nullptr));
        actionUndo->setText(QCoreApplication::translate("MainWindow", "&Undo", nullptr));
        actionRedo->setText(QCoreApplication::translate("MainWindow", "&Redo", nullptr));
        actionFirst->setText(QCoreApplication::translate("MainWindow", "&First", nullptr));
        actionGrayscale->setText(QCoreApplication::translate("MainWindow", "&Grayscale", nullptr));
        actionGaussianBlur->setText(QCoreApplication::translate("MainWindow", "&Gaussian Blur", nullptr));
        actionCannyEdges->setText(QCoreApplication::translate("MainWindow", "&Canny Edges", nullptr));
        actionMedianFilter->setText(QCoreApplication::translate("MainWindow", "&Median Filter", nullptr));
        actionSobelFilter->setText(QCoreApplication::translate("MainWindow", "&Sobel Filter", nullptr));
        actionLaplacianFilter->setText(QCoreApplication::translate("MainWindow", "&Laplacian Filter", nullptr));
        actionBilateralFilter->setText(QCoreApplication::translate("MainWindow", "&Bilateral Filter", nullptr));
        label_opencv->setText(QCoreApplication::translate("MainWindow", "label_opencv", nullptr));
        label_ipp->setText(QCoreApplication::translate("MainWindow", "label_ipp", nullptr));
        label_opencv_title->setText(QCoreApplication::translate("MainWindow", "Processing Time:", nullptr));
        label_ipp_title->setText(QCoreApplication::translate("MainWindow", "Processing Time:", nullptr));
        label_cuda->setText(QCoreApplication::translate("MainWindow", "label_cuda", nullptr));
        label_cudakernel->setText(QCoreApplication::translate("MainWindow", "label_cudakernel", nullptr));
        label_cuda_title->setText(QCoreApplication::translate("MainWindow", "Processing Time:", nullptr));
        label_cudakernel_title->setText(QCoreApplication::translate("MainWindow", "Processing Time:", nullptr));
        menuFile->setTitle(QCoreApplication::translate("MainWindow", "File", nullptr));
        menuEdit->setTitle(QCoreApplication::translate("MainWindow", "Edit", nullptr));
        menuFilters->setTitle(QCoreApplication::translate("MainWindow", "Filters", nullptr));
        (void)MainWindow;
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
