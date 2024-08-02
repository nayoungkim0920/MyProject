// imageProcessingLib.cpp
#include "imageProcessingLib.h"
#include "pch.h"
// 그레이스케일 이미지 버퍼 크기 계산

Ipp8u* matToIpp8u(cv::Mat& mat)
{
    return mat.ptr<Ipp8u>();
}

// 에러 체크 함수
void checkCudaError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// NPP 오류 처리 함수
void checkNPPError(NppStatus status) {
    if (status != NPP_SUCCESS) {
        std::cerr << "NPP 오류: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

//void checkNppError(NppStatus status, const std::string& errorMessage) {
//    if (status != NPP_SUCCESS) {
//        std::cerr << errorMessage << " Error code: " << status << std::endl;
//        throw std::runtime_error(errorMessage);
//    }
//}

void printImagePixels(cv::Mat& image, int numPixels) { //0:전체
    int count = 0;

    // 이미지의 크기를 가져옵니다.
    int rows = image.rows;
    int cols = image.cols;

    // 이미지가 비어있거나 픽셀 수가 0보다 작으면 반환합니다.
    if (rows == 0 || cols == 0) {
        std::cerr << "이미지 크기가 잘못되었거나 이미지가 비어 있습니다." << std::endl;
        return;
    }

    // numPixels가 0인 경우 전체 픽셀을 출력하도록 설정합니다.
    if (numPixels <= 0) {
        numPixels = rows * cols;
    }

    // 각 픽셀을 출력합니다.
    for (int y = 0; y < rows && count < numPixels; ++y) {
        for (int x = 0; x < cols && count < numPixels; ++x) {
            // 픽셀 값을 가져옵니다.
            if (image.channels() == 3) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
                std::cout << "Pixel (" << x << ", " << y << "): "
                    << "B=" << static_cast<int>(pixel[0]) << ", "
                    << "G=" << static_cast<int>(pixel[1]) << ", "
                    << "R=" << static_cast<int>(pixel[2]) << std::endl;
            }
            else if (image.channels() == 1) {
                uchar pixel = image.at<uchar>(y, x);
                std::cout << "Pixel (" << x << ", " << y << "): "
                    << "Gray=" << static_cast<int>(pixel) << std::endl;
            }

            count++;
        }
    }

    if (count == 0) {
        std::cerr << "픽셀을 출력할 수 없습니다." << std::endl;
    }
}

// cv::Mat을 NPP 이미지로 변환
Npp8u* matToNppImage(cv::Mat& mat, NppiSize& size, int& nppSize) {
    nppSize = mat.cols * mat.rows * mat.elemSize();
    Npp8u* pNppImage = new Npp8u[nppSize];
    memcpy(pNppImage, mat.data, nppSize);
    size.width = mat.cols;
    size.height = mat.rows;
    return pNppImage;
}

// NPP 이미지에서 cv::Mat으로 변환
cv::Mat nppImageToMat(Npp8u* pNppImage, NppiSize size, int nppSize) {
    cv::Mat mat(size.height, size.width, CV_8UC3, pNppImage);
    return mat;
}

std::unordered_map<int, QString> typeToStringMap = {
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
    // 추가적인 이미지 타입에 대한 설명을 필요에 따라 추가할 수 있습니다.
};

// 정적 메서드 구현
QString getImageTypeString(int type) {
    if (typeToStringMap.find(type) != typeToStringMap.end()) {
        return typeToStringMap[type];
    }
    else {
        return "Unknown type";
    }
}

void checkNPPStatus(NppStatus status, const std::string& context) {
    if (status != NPP_SUCCESS) {
        std::cerr << "NPP error in " << context << ": " << status << std::endl;
        switch (status) {
        case NPP_CUDA_KERNEL_EXECUTION_ERROR:
            std::cerr << "CUDA kernel execution error." << std::endl;
            break;
        case NPP_BAD_ARGUMENT_ERROR:
            std::cerr << "Bad argument error." << std::endl;
            break;
        case NPP_MEMORY_ALLOCATION_ERR:
            std::cerr << "Memory allocation error." << std::endl;
            break;
        default:
            std::cerr << "Unknown error: " << status << std::endl;
            break;
        }
    }
}

void checkDeviceProperties() {
    int deviceCount;
    cudaError_t cudaErr = cudaGetDeviceCount(&deviceCount);
    if (cudaErr != cudaSuccess) {
        std::cerr << "Failed to get device count: " << cudaGetErrorString(cudaErr) << std::endl;
        return;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaErr = cudaGetDeviceProperties(&deviceProp, i);
        if (cudaErr != cudaSuccess) {
            std::cerr << "Failed to get device properties: " << cudaGetErrorString(cudaErr) << std::endl;
            return;
        }

        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Max Threads Per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Grid Size: (" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
    }
}

cv::Mat gstBufferToMat(GstBuffer* buffer, GstCaps* caps) {
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_READ);

    gint width, height;
    const gchar* format;
    GstStructure* structure = gst_caps_get_structure(caps, 0);

    gst_structure_get_int(structure, "width", &width);
    gst_structure_get_int(structure, "height", &height);

    format = gst_structure_get_string(structure, "format");
    if (!format) {
        std::cerr << "비디오 포맷을 가져오는 데 실패했습니다." << std::endl;
        gst_buffer_unmap(buffer, &map);
        return cv::Mat();
    }

    cv::Mat mat;
    if (strcmp(format, "BGR") == 0) {
        mat = cv::Mat(height, width, CV_8UC3, map.data, map.size / height).clone();
    }
    else if (strcmp(format, "GRAY8") == 0) {
        mat = cv::Mat(height, width, CV_8UC1, map.data, map.size / height).clone();
    }
    else {
        std::cerr << "지원하지 않는 비디오 포맷입니다: " << format << std::endl;
    }

    gst_buffer_unmap(buffer, &map);
    return mat;
}

// cv::Mat을 GstBuffer로 변환
GstBuffer* matToGstBuffer(cv::Mat& mat) {
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, mat.total() * mat.elemSize() + 1000, nullptr);
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    std::memcpy(map.data, mat.data, mat.total() * mat.elemSize());
    gst_buffer_unmap(buffer, &map);
    return buffer;
}

void drawEdgesOnColorImage(cv::Mat& image, const cv::Mat& edges) {
    // 엣지 이미지를 컬러 이미지에 초록색으로 오버레이
    for (int y = 0; y < edges.rows; ++y) {
        for (int x = 0; x < edges.cols; ++x) {
            if (edges.at<uchar>(y, x) == 255) {
                image.at<cv::Vec3b>(y, x)[0] = 0;   // Blue channel
                image.at<cv::Vec3b>(y, x)[1] = 255; // Green channel
                image.at<cv::Vec3b>(y, x)[2] = 0;   // Red channel
            }
        }
    }
}

void drawEdgesOnGrayImage(cv::Mat& image, const cv::Mat& edges) {
    // 엣지 이미지를 그레이스케일 이미지에 흰색으로 오버레이
    for (int y = 0; y < edges.rows; ++y) {
        for (int x = 0; x < edges.cols; ++x) {
            if (edges.at<uchar>(y, x) == 255) {
                image.at<uchar>(y, x) = 255;
            }
            else {
                image.at<uchar>(y, x) = 0;
            }
        }
    }
}

// NPP 파라미터 출력 함수
void printNppParameters(const std::string& label, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, const Npp8u* pSrc, const Npp8u* pDst, const Npp8u* pBuffer) {
    std::cout << label << " Parameters:" << std::endl;
    std::cout << "ROI Size: (" << oSizeROI.width << ", " << oSizeROI.height << ")" << std::endl;
    std::cout << "Mask Size: (" << oMaskSize.width << ", " << oMaskSize.height << ")" << std::endl;
    std::cout << "Anchor: (" << oAnchor.x << ", " << oAnchor.y << ")" << std::endl;
    std::cout << "Source Pointer: " << static_cast<void*>(const_cast<Npp8u*>(pSrc)) << std::endl;
    std::cout << "Destination Pointer: " << static_cast<void*>(const_cast<Npp8u*>(pDst)) << std::endl;
    std::cout << "Buffer Pointer: " << static_cast<void*>(const_cast<Npp8u*>(pBuffer)) << std::endl;
}

void printBufferSize(const std::string& prefix, Npp32u bufferSize) {
    std::cout << prefix << " Buffer Size: " << bufferSize << std::endl;
}