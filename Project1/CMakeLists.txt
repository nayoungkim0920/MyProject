cmake_minimum_required(VERSION 3.14)
project(Project1 LANGUAGES CXX CUDA)

# Qt, OpenCV, CUDA 설정
set(CMAKE_PREFIX_PATH "C:/Qt/6.7.1/msvc2019_64" "C:/opencv/build")
find_package(Qt6 REQUIRED COMPONENTS Widgets Core Gui)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui cudaarithm cudafilters cudawarping cudacodec cudafeatures2d cudaimgproc)
find_package(CUDA REQUIRED)

# LibTorch 설정
set(TORCH_ROOT "C:/libtorch")
#if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#    set(TORCH_ROOT "C:/myLab/libtorch_Debug")
#else()
#    set(TORCH_ROOT "C:/myLab/libtorch_Release")
#endif()

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${TORCH_ROOT}")
find_package(Torch REQUIRED)

# IPP 설정
set(IPP_ROOT "C:/Program Files (x86)/Intel/oneAPI/ipp/2021.11")
include_directories("${IPP_ROOT}/include")
link_directories("${IPP_ROOT}/lib")

# GStreamer 설정
set(GSTREAMER_ROOT "C:/gstreamer/1.0/msvc_x86_64")
include_directories(
    "${GSTREAMER_ROOT}/include/gstreamer-1.0"
    "${GSTREAMER_ROOT}/include/glib-2.0"
    "${GSTREAMER_ROOT}/lib/glib-2.0/include"
)
link_directories("${GSTREAMER_ROOT}/lib")

# IPP 설정
set(IPLIB_ROOT "C:/myLab/Project1/Project1/imageProcessingLib")
include_directories("${IPLIB_ROOT}")
link_directories(
    "${IPLIB_ROOT}/build/lib/Debug"
    "${IPLIB_ROOT}/build/bin/Debug"
    "${IPLIB_ROOT}/build/Debug"
    )

# 추가 포함 디렉터리 설정
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    "${CUDA_TOOLKIT_ROOT_DIR}/include"
    ${IPP_ROOT}/include
    "${GSTREAMER_ROOT}/include/gstreamer-1.0"
    "${GSTREAMER_ROOT}/lib/gstreamer-1.0/include"
    ${TORCH_ROOT}/include  # LibTorch 포함 디렉토리
)

# CUDA 파일 설정
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14 --expt-relaxed-constexpr -gencode arch=compute_86,code=sm_86)

# Qt 래핑 파일 생성
qt6_wrap_cpp(MOC_FILES
    MainWindow.h
    ImageProcessor.h
    yolov5dialog.h
)

# CUDA 파일 컴파일 및 라이브러리 생성
cuda_add_library(image_processing
    imageProcessing.cu
)

# 라이브러리 디렉토리 설정
link_directories("${CMAKE_BINARY_DIR}")  # 라이브러리 디렉토리 설정

# 라이브러리 링크
target_link_libraries(image_processing
    ippcc.lib
    ippcore.lib
    ippvm.lib
    ipps.lib
    ippi.lib
    ippcv.lib
    nppial
    nppicc
    nppidei
    nppif
    nppig
    nppim
    nppist
    nppisu
    nppitc
    npps
    imageProcessingLib
)

# GStreamer 라이브러리 추가
list(APPEND GSTREAMER_LIBRARIES
    gstreamer-1.0.lib
    gobject-2.0.lib
    glib-2.0.lib
    gstapp-1.0.lib
    gstbase-1.0.lib
    gstvideo-1.0.lib
)

# 실행 파일 추가
add_executable(Project1
    main.cpp
    MainWindow.cpp
    MainWindow.h
    MainWindow.ui
    ImageProcessor.cpp
    ImageProcessor.h
    ${MOC_FILES}
)

# 라이브러리 링크
target_link_libraries(Project1
    Qt6::Widgets
    Qt6::Core
    Qt6::Gui
    ${OpenCV_LIBS}
    image_processing
    imageProcessingLib
    ${CUDA_LIBRARIES}
    ${CUDNN_LIBRARIES}
    ${GSTREAMER_LIBRARIES}
    ${TORCH_ROOT}/lib # LibTorch 링크
)

# 빌드 후 TARGET_FILE_DIR 출력
add_custom_command(TARGET Project1 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "TARGET_FILE_DIR for Project1: $<TARGET_FILE_DIR:Project1>"
    COMMAND ${CMAKE_COMMAND} -E echo "$<TARGET_FILE_DIR:Project1>" > "Project1_dir.txt"
)

# 실행 파일 출력 디렉토리 설정
set_target_properties(Project1 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release"
)

file(COPY "C:/myLab/Project1/Project1/imageProcessingLib/build/Debug/imageProcessingLib.dll"
     DESTINATION "${CMAKE_BINARY_DIR}/Debug"
)

#file(COPY "C:/myLab/Project1/Project1/imageProcessingLib/build/bin/Debug/imageProcessingLib.lib"
#     DESTINATION "${CMAKE_BINARY_DIR}/Debug"
#)

#file(COPY "C:/myLab/Project1/Project1/imageProcessingLib/build/bin/Debug/imageProcessingLib.lib"
#     DESTINATION "${CMAKE_BINARY_DIR}/Debug"
#)

# 디버그 빌드에서의 OpenCV opencv_world DLL 복사
add_custom_command(TARGET Project1 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "C:/opencv/build/x64/vc16/bin/opencv_world4100d.dll" "$<TARGET_FILE_DIR:Project1>"
    COMMENT "Copying OpenCV Debug opencv_world DLL to output directory"
)

# 릴리스 빌드에서의 OpenCV opencv_world DLL 복사
add_custom_command(TARGET Project1 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "C:/opencv/build/x64/vc16/bin/opencv_world4100.dll" "$<TARGET_FILE_DIR:Project1>"
    COMMENT "Copying OpenCV Release opencv_world DLL to output directory"
)

# 디버그 빌드에서의 OpenCV DLL 복사
add_custom_command(TARGET Project1 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "C:/opencv/build/bin/Debug/opencv_cudaarithm4100d.dll" "$<TARGET_FILE_DIR:Project1>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "C:/opencv/build/bin/Debug/opencv_cudaimgproc4100d.dll" "$<TARGET_FILE_DIR:Project1>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "C:/opencv/build/bin/Debug/opencv_cudafilters4100d.dll" "$<TARGET_FILE_DIR:Project1>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "C:/opencv/build/bin/Debug/opencv_imgcodecs4100d.dll" "$<TARGET_FILE_DIR:Project1>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "C:/opencv/build/bin/Debug/opencv_core4100d.dll" "$<TARGET_FILE_DIR:Project1>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "C:/opencv/build/bin/Debug/opencv_imgproc4100d.dll" "$<TARGET_FILE_DIR:Project1>"
    COMMENT "Copying OpenCV Debug DLLs to output directory"
)

# 릴리스 빌드에서의 OpenCV DLL 복사
add_custom_command(TARGET Project1 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "C:/opencv/build/bin/Release/opencv_cudaarithm4100.dll" "$<TARGET_FILE_DIR:Project1>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "C:/opencv/build/bin/Release/opencv_cudaimgproc4100.dll" "$<TARGET_FILE_DIR:Project1>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "C:/opencv/build/bin/Release/opencv_cudafilters4100.dll" "$<TARGET_FILE_DIR:Project1>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "C:/opencv/build/bin/Release/opencv_imgcodecs4100.dll" "$<TARGET_FILE_DIR:Project1>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "C:/opencv/build/bin/Release/opencv_core4100.dll" "$<TARGET_FILE_DIR:Project1>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "C:/opencv/build/bin/Release/opencv_imgproc4100.dll" "$<TARGET_FILE_DIR:Project1>"
    COMMENT "Copying OpenCV Release DLLs to output directory"
)

# 파일 인코딩 설정 추가
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

# OpenMP 설정
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Project1 OpenMP::OpenMP_CXX)
endif()

if (MSVC)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Project1)

    # Debug 빌드
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MDd")

    # Release 빌드
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MD")

    # 반복자 디버그 레벨 설정
    add_compile_definitions(
        $<$<CONFIG:Debug>:_ITERATOR_DEBUG_LEVEL=2>
        $<$<CONFIG:Release>:_ITERATOR_DEBUG_LEVEL=0>
    )
endif()