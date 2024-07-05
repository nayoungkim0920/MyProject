cmake_minimum_required(VERSION 3.14)
project(Project1 LANGUAGES CXX CUDA)

# Qt, OpenCV, CUDA 설정
set(CMAKE_PREFIX_PATH "C:/Qt/6.7.1/msvc2019_64" "C:/opencv/build")
find_package(Qt6 REQUIRED COMPONENTS Widgets Core Gui)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui cudaarithm cudafilters cudawarping cudacodec cudafeatures2d cudaimgproc)
find_package(CUDA REQUIRED)

# 추가 포함 디렉터리 설정
include_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${CUDNN_ROOT_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# CUDA 파일 설정
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14 --expt-relaxed-constexpr -gencode arch=compute_86,code=sm_86)

# Qt 래핑 파일 생성
qt6_wrap_cpp(MOC_FILES
    MainWindow.h
    ImageProcessor.h
)

# CUDA 파일 컴파일 및 라이브러리 생성
cuda_add_library(image_processing
    imageProcessing.cu
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
    ${CUDA_LIBRARIES}
    ${CUDNN_LIBRARIES}
    ${CUDA_cudart_LIBRARY}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_curand_LIBRARY}
)

# 실행 파일 출력 디렉토리 설정
set_target_properties(Project1 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release"
)

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

# IPP 헤더 파일 포함
target_include_directories(Project1 PRIVATE
    ${IPP_INCLUDE_DIR}
)

# 파일 인코딩 설정 추가
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

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
