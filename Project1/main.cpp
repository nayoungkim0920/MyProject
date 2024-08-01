//main.cpp
#include "MainWindow.h"
#include <QApplication>
#include <cstdlib>

int main(int argc, char* argv[])
{
    putenv("GST_DEBUG=3");
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}