cmake_minimum_required(VERSION 3.14)
project(imageProcessingLib LANGUAGES CXX)

# 필요한 패키지 찾기
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Core)  # Qt6 Core 모듈 추가

# NPP 및 IPP 라이브러리 설정
set(NPP_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5")
set(IPP_ROOT "C:/Program Files (x86)/Intel/oneAPI/ipp/2021.11")

# GStreamer 설정
set(GSTREAMER_ROOT "C:/gstreamer/1.0/msvc_x86_64")

# 라이브러리의 소스 파일 추가
add_library(imageProcessingLib SHARED
    imageProcessingLib.cpp
    imageProcessingLib.h
)

# 필요한 디렉토리 설정
target_include_directories(imageProcessingLib PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${NPP_ROOT}/include
    ${IPP_ROOT}/include
    ${GSTREAMER_ROOT}/include/gstreamer-1.0
    ${GSTREAMER_ROOT}/include/glib-2.0
    ${GSTREAMER_ROOT}/lib/glib-2.0/include
    ${Qt6_INCLUDE_DIRS}  # Qt6 포함 디렉토리 추가
)

# 라이브러리 링크
target_link_libraries(imageProcessingLib
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${NPP_ROOT}/lib/x64/nppc.lib
    ${NPP_ROOT}/lib/x64/npps.lib
    ${IPP_ROOT}/lib/ippcore.lib
    ${IPP_ROOT}/lib/ippvm.lib
    ${IPP_ROOT}/lib/ippcc.lib
    ${GSTREAMER_ROOT}/lib/gstreamer-1.0.lib
    ${GSTREAMER_ROOT}/lib/gobject-2.0.lib
    ${GSTREAMER_ROOT}/lib/glib-2.0.lib
    ${GSTREAMER_ROOT}/lib/gstapp-1.0.lib
    ${GSTREAMER_ROOT}/lib/gstbase-1.0.lib
    ${GSTREAMER_ROOT}/lib/gstvideo-1.0.lib
    Qt6::Core  # Qt6 Core 라이브러리 링크
)

# 라이브러리의 출력 디렉토리 설정
set_target_properties(imageProcessingLib PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"  # 정적 라이브러리의 .lib 파일
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"  # 동적 라이브러리의 .dll 파일
)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
