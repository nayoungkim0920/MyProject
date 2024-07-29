#ifndef IMAGEPROCESSINGLIB_H
#define IMAGEPROCESSINGLIB_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <ipp.h>
#include <ipp/ippbase.h>
#include <npp.h>
#include <nppi.h>
#include <gst/gst.h>
#include <QString>
#include <nppi_filtering_functions.h>
#include <iostream>
#include <unordered_map>

#include "pch.h"

// EXPORT 매크로 정의 (Windows용)
#ifdef IMAGEPROCESSINGLIB_EXPORTS
#define IMAGEPROCESSINGLIB_API __declspec(dllexport)
#else
#define IMAGEPROCESSINGLIB_API __declspec(dllimport)
#endif

// C++ 인터페이스 함수 선언
//IMAGEPROCESSINGLIB_API void calculateGrayScaleBufferSize(const NppiSize& oSizeROI, const NppiSize& oMaskSize, Npp32u& nBufferSize);
//IMAGEPROCESSINGLIB_API void calculateBufferSize(const NppiSize& oSizeROI, const NppiSize& oMaskSize, Npp32u& nBufferSize);
IMAGEPROCESSINGLIB_API Ipp8u* matToIpp8u(cv::Mat& mat);
IMAGEPROCESSINGLIB_API void checkNPPError(NppStatus status);
IMAGEPROCESSINGLIB_API void printImagePixels(cv::Mat& image, int numPixels);
IMAGEPROCESSINGLIB_API Npp8u* matToNppImage(cv::Mat& mat, NppiSize& size, int& nppSize);
IMAGEPROCESSINGLIB_API cv::Mat nppImageToMat(Npp8u* pNppImage, NppiSize size, int nppSize);
IMAGEPROCESSINGLIB_API QString getImageTypeString(int type);
IMAGEPROCESSINGLIB_API void checkNPPStatus(NppStatus status, const std::string& context);
IMAGEPROCESSINGLIB_API void checkCudaError(const char* msg);
IMAGEPROCESSINGLIB_API void checkDeviceProperties();
IMAGEPROCESSINGLIB_API GstBuffer* matToGstBuffer(cv::Mat& mat);
IMAGEPROCESSINGLIB_API cv::Mat gstBufferToMat(GstBuffer* buffer, GstCaps* caps);
IMAGEPROCESSINGLIB_API void drawEdgesOnColorImage(cv::Mat& image, const cv::Mat& edges);
IMAGEPROCESSINGLIB_API void drawEdgesOnGrayImage(cv::Mat& image, const cv::Mat& edges);
IMAGEPROCESSINGLIB_API void printNppParameters(const std::string& label, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, const Npp8u* pSrc, const Npp8u* pDst, const Npp8u* pBuffer);
IMAGEPROCESSINGLIB_API void printBufferSize(const std::string& prefix, Npp32u bufferSize);

#endif // IMAGEPROCESSINGLIB_H