//ImageTypeConverter.cpp
#include "ImageTypeConverter.h"

// ���� ��� ���� ����
std::unordered_map<int, QString> ImageTypeConverter::typeToStringMap = {
    {CV_8UC1, "CV_8UC1 8-bit single-channel (grayscale)"},
    {CV_8UC2, "CV_8UC2 8-bit 2-channel"},
    {CV_8UC3, "CV_8UC3 8-bit 3-channel (BGR)"},
    {CV_8UC4, "CV_8UC4 8-bit 4-channel"},
    {CV_16UC1, "CV_16UC1 16-bit single-channel"},
    {CV_16UC3, "CV_16UC3 16-bit 3-channel"},
    {CV_32FC1, "CV_32FC1 32-bit single-channel (float)"},
    {CV_32FC3, "CV_32FC3 32-bit 3-channel (float)"},
    {CV_16SC1, "CV_16SC1 16-bit 1-channel"},
    {CV_16SC3, "CV_16SC3 16-bit 3-channel"}
    // �߰����� �̹��� Ÿ�Կ� ���� ������ �ʿ信 ���� �߰��� �� �ֽ��ϴ�.
};

// ���� �޼��� ����
QString ImageTypeConverter::getImageTypeString(int type) {
    if (typeToStringMap.find(type) != typeToStringMap.end()) {
        return typeToStringMap[type];
    }
    else {
        return "Unknown type";
    }
}