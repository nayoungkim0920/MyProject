//ImageTypeConverter.h
#ifndef IMAGETYPECONVERTER_H
#define IMAGETYPECONVERTER_H

#include <QString>
#include <unordered_map>
#include <opencv2/opencv.hpp>

class ImageTypeConverter {
public:
    // ���� ��� ���� ����
    static std::unordered_map<int, QString> typeToStringMap;

    // ���� �޼��� ����
    static QString getImageTypeString(int type);
};

#endif // IMAGETYPECONVERTER_H