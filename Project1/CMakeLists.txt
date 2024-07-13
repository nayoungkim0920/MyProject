cmake_minimum_required(VERSION 3.14)
project(Project1 LANGUAGES CXX CUDA)

# Qt, OpenCV, CUDA 설정
set(CMAKE_PREFIX_PATH "C:/Qt/6.7.1/msvc2019_64" "C:/opencv/build")
find_package(Qt6 REQUIRED COMPONENTS Widgets Core Gui)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui cudaarithm cudafilters cudawarping cudacodec cudafeatures2d cudaimgproc)
find_package(CUDA REQUIRED)

# CUDA 경로 설정
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5")
find_package(CUDA REQUIRED)

# IPP 설정
set(IPP_ROOT "C:/Program Files (x86)/Intel/oneAPI/ipp/2021.11")  # IPP 라이브러리 설치 경로
include_directories("${IPP_ROOT}/include")
link_directories("${IPP_ROOT}/lib")  # IPP 라이브러리 lib 경로

# 추가 포함 디렉터리 설정
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    "${CUDA_TOOLKIT_ROOT_DIR}/include"  # CUDA Toolkit include 폴더 추가
    ${IPP_ROOT}/include  # IPP include 폴더 추가
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

# IPP 라이브러리 링크
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
)

# 실행 파일 출력 디렉토리 설정
set_target_properties(Project1 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release"
)

# 파일 인코딩 설정 추가
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

# OpenMP 설정
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Project1 
    OpenMP::OpenMP_CXX)
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
