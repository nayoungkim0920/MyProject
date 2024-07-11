//ImageTypeConverter.h
#ifndef IMAGETYPECONVERTER_H
#define IMAGETYPECONVERTER_H

#include <QString>
#include <unordered_map>
#include <opencv2/opencv.hpp>

class ImageTypeConverter {
public:
    // 정적 멤버 변수 선언
    static std::unordered_map<int, QString> typeToStringMap;

    // 정적 메서드 선언
    static QString getImageTypeString(int type);
};

#endif // IMAGETYPECONVERTER_H