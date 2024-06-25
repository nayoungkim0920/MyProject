/********************************************************************************
** Form generated from reading UI file 'MainWIndow.ui'
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
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
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
    QAction *actionGrayscale;
    QAction *actionGaussianBlur;
    QAction *actionCannyEdges;
    QAction *actionMedianFilter;
    QAction *actionSobelFilter;
    QAction *actionLaplacianFilter;
    QAction *actionBilateralFilter;
    QAction *actionUndo;
    QAction *actionRedo;
    QAction *actionFirst;
    QWidget *centralwidget;
    QVBoxLayout *verticalLayout;
    QLabel *label;
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
        actionUndo = new QAction(MainWindow);
        actionUndo->setObjectName("actionUndo");
        actionRedo = new QAction(MainWindow);
        actionRedo->setObjectName("actionRedo");
        actionFirst = new QAction(MainWindow);
        actionFirst->setObjectName("actionFirst");
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName("centralwidget");
        verticalLayout = new QVBoxLayout(centralwidget);
        verticalLayout->setObjectName("verticalLayout");
        label = new QLabel(centralwidget);
        label->setObjectName("label");
        label->setAlignment(Qt::AlignCenter);

        verticalLayout->addWidget(label);

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
        actionOpen->setText(QCoreApplication::translate("MainWindow", "Open", nullptr));
        actionSave->setText(QCoreApplication::translate("MainWindow", "Save", nullptr));
        actionExit->setText(QCoreApplication::translate("MainWindow", "Exit", nullptr));
        actionRotate->setText(QCoreApplication::translate("MainWindow", "Rotate", nullptr));
        actionZoomIn->setText(QCoreApplication::translate("MainWindow", "Zoom In", nullptr));
        actionZoomOut->setText(QCoreApplication::translate("MainWindow", "Zoom Out", nullptr));
        actionGrayscale->setText(QCoreApplication::translate("MainWindow", "Grayscale", nullptr));
        actionGaussianBlur->setText(QCoreApplication::translate("MainWindow", "Gaussian", nullptr));
        actionCannyEdges->setText(QCoreApplication::translate("MainWindow", "Canny", nullptr));
        actionMedianFilter->setText(QCoreApplication::translate("MainWindow", "Median", nullptr));
        actionSobelFilter->setText(QCoreApplication::translate("MainWindow", "Sobel", nullptr));
        actionLaplacianFilter->setText(QCoreApplication::translate("MainWindow", "Laplacian", nullptr));
        actionBilateralFilter->setText(QCoreApplication::translate("MainWindow", "Bilateral", nullptr));
        actionUndo->setText(QCoreApplication::translate("MainWindow", "Undo", nullptr));
#if QT_CONFIG(shortcut)
        actionUndo->setShortcut(QCoreApplication::translate("MainWindow", "Ctrl+Z", nullptr));
#endif // QT_CONFIG(shortcut)
        actionRedo->setText(QCoreApplication::translate("MainWindow", "Redo", nullptr));
#if QT_CONFIG(shortcut)
        actionRedo->setShortcut(QCoreApplication::translate("MainWindow", "Ctrl+Y", nullptr));
#endif // QT_CONFIG(shortcut)
        actionFirst->setText(QCoreApplication::translate("MainWindow", "First", nullptr));
        label->setText(QCoreApplication::translate("MainWindow", "Image Display", nullptr));
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
