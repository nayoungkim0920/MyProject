cmake_minimum_required(VERSION 3.14)
project(Project1)

# Qt, OpenCV, CUDA 설정
set(CMAKE_PREFIX_PATH "C:/Qt/6.7.1/msvc2019_64" "C:/opencv/build")
find_package(Qt6 REQUIRED COMPONENTS Widgets Core Gui)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui cudaarithm cudafilters cudawarping cudacodec cudafeatures2d cudaimgproc)

# CUDA 및 cuDNN 경로 설정
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1")
set(CUDNN_ROOT_DIR "C:/Program Files/NVIDIA/CUDNN/v9.1")

# 추가 포함 디렉터리 설정
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIRS}
)

# CUDA 파일 설정
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 --expt-relaxed-constexpr)

# 실행 파일 추가
qt6_wrap_cpp(MOC_FILES
    MainWindow.h
    ImageProcessor.h
)

add_executable(Project1
    main.cpp
    MainWindow.cpp
    MainWindow.h
    MainWindow.ui
    ImageProcessor.cpp
    ImageProcessor.h
    ${MOC_FILES}  # 이전에 생성한 MOC 파일
)

# Qt 및 OpenCV 라이브러리 링크
target_link_libraries(Project1
    Qt6::Widgets
    Qt6::Core
    Qt6::Gui
    ${OpenCV_LIBS}
)

# CUDA 및 cuDNN 라이브러리 링크
target_link_libraries(Project1
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
# Visual Studio의 기본 인코딩을 UTF-8로 설정
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 한글 주석 예시
# 아래와 같은 주석을 사용하여 코드의 의도를 설명할 수 있습니다.
# 이 주석은 한글이 포함된 주석입니다. GitHub에 업로드할 때 인코딩 문제가 발생하지 않도록 합니다.

