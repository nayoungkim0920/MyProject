/****************************************************************************
** Meta object code from reading C++ file 'MainWindow.h'
**
** Created by: The Qt Meta Object Compiler version 68 (Qt 6.7.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../MainWindow.h"
#include <QtCore/qmetatype.h>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'MainWindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 68
#error "This file was generated using the moc from 6.7.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

#ifndef Q_CONSTINIT
#define Q_CONSTINIT
#endif

QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
QT_WARNING_DISABLE_GCC("-Wuseless-cast")
namespace {

#ifdef QT_MOC_HAS_STRINGDATA
struct qt_meta_stringdata_CLASSMainWindowENDCLASS_t {};
constexpr auto qt_meta_stringdata_CLASSMainWindowENDCLASS = QtMocHelpers::stringData(
    "MainWindow",
    "openFile",
    "",
    "saveFile",
    "rotateImage",
    "zoomInImage",
    "zoomOutImage",
    "convertToGrayscale",
    "applyGaussianBlur",
    "cannyEdges",
    "medianFilter",
    "laplacianFilter",
    "bilateralFilter",
    "exitApplication",
    "redoAction",
    "undoAction",
    "first",
    "displayImage",
    "cv::Mat",
    "image"
);
#else  // !QT_MOC_HAS_STRINGDATA
#error "qtmochelpers.h not found or too old."
#endif // !QT_MOC_HAS_STRINGDATA
} // unnamed namespace

Q_CONSTINIT static const uint qt_meta_data_CLASSMainWindowENDCLASS[] = {

 // content:
      12,       // revision
       0,       // classname
       0,    0, // classinfo
      16,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags, initial metatype offsets
       1,    0,  110,    2, 0x08,    1 /* Private */,
       3,    0,  111,    2, 0x08,    2 /* Private */,
       4,    0,  112,    2, 0x08,    3 /* Private */,
       5,    0,  113,    2, 0x08,    4 /* Private */,
       6,    0,  114,    2, 0x08,    5 /* Private */,
       7,    0,  115,    2, 0x08,    6 /* Private */,
       8,    0,  116,    2, 0x08,    7 /* Private */,
       9,    0,  117,    2, 0x08,    8 /* Private */,
      10,    0,  118,    2, 0x08,    9 /* Private */,
      11,    0,  119,    2, 0x08,   10 /* Private */,
      12,    0,  120,    2, 0x08,   11 /* Private */,
      13,    0,  121,    2, 0x08,   12 /* Private */,
      14,    0,  122,    2, 0x08,   13 /* Private */,
      15,    0,  123,    2, 0x08,   14 /* Private */,
      16,    0,  124,    2, 0x08,   15 /* Private */,
      17,    1,  125,    2, 0x08,   16 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 18,   19,

       0        // eod
};

Q_CONSTINIT const QMetaObject MainWindow::staticMetaObject = { {
    QMetaObject::SuperData::link<QMainWindow::staticMetaObject>(),
    qt_meta_stringdata_CLASSMainWindowENDCLASS.offsetsAndSizes,
    qt_meta_data_CLASSMainWindowENDCLASS,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_stringdata_CLASSMainWindowENDCLASS_t,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<MainWindow, std::true_type>,
        // method 'openFile'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'saveFile'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'rotateImage'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'zoomInImage'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'zoomOutImage'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'convertToGrayscale'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'applyGaussianBlur'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'cannyEdges'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'medianFilter'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'laplacianFilter'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'bilateralFilter'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'exitApplication'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'redoAction'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'undoAction'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'first'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'displayImage'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const cv::Mat &, std::false_type>
    >,
    nullptr
} };

void MainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<MainWindow *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->openFile(); break;
        case 1: _t->saveFile(); break;
        case 2: _t->rotateImage(); break;
        case 3: _t->zoomInImage(); break;
        case 4: _t->zoomOutImage(); break;
        case 5: _t->convertToGrayscale(); break;
        case 6: _t->applyGaussianBlur(); break;
        case 7: _t->cannyEdges(); break;
        case 8: _t->medianFilter(); break;
        case 9: _t->laplacianFilter(); break;
        case 10: _t->bilateralFilter(); break;
        case 11: _t->exitApplication(); break;
        case 12: _t->redoAction(); break;
        case 13: _t->undoAction(); break;
        case 14: _t->first(); break;
        case 15: _t->displayImage((*reinterpret_cast< std::add_pointer_t<cv::Mat>>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObject *MainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_CLASSMainWindowENDCLASS.stringdata0))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 16)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 16;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 16)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 16;
    }
    return _id;
}
QT_WARNING_POP
