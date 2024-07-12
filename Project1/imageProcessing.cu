//imageProecssing.cu
#include "imageProcessing.cuh"

__device__ void rotatePixel(int x, int y, int cols, int rows, int channels, const unsigned char* input, unsigned char* output) {
    if (x < cols && y < rows) {
        for (int c = 0; c < channels; ++c) {
            output[(x * rows + (rows - 1 - y)) * channels + c] = input[(y * cols + x) * channels + c];
        }
    }
}

__device__ float gaussian(float x, float sigma) {
    return expf(-(x * x) / (2 * sigma * sigma));
}

__global__ void rotateImageKernel(const unsigned char* input, unsigned char* output, int cols, int rows, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        rotatePixel(x, y, cols, rows, channels, input, output);
    }
}

__global__ void resizeImageKernel(const unsigned char* input, unsigned char* output, int oldWidth, int oldHeight, int newWidth, int newHeight, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < newWidth && y < newHeight) {
        float x_ratio = oldWidth / (float)newWidth;
        float y_ratio = oldHeight / (float)newHeight;
        int px = floor(x * x_ratio);
        int py = floor(y * y_ratio);

        for (int c = 0; c < channels; ++c) {
            output[(y * newWidth + x) * channels + c] = input[(py * oldWidth + px) * channels + c];
        }
    }
}

__global__ void grayScaleImageKernel(const unsigned char* input, unsigned char* output, int cols, int rows) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int idx = y * cols + x;
        unsigned char b = input[idx * 3 + 0];
        unsigned char g = input[idx * 3 + 1];
        unsigned char r = input[idx * 3 + 2];
        output[idx] = 0.299f * r + 0.587f * g + 0.114f * b;  // 그레이스케일 변환
    }
}

__global__ void cannyEdgesKernel(const unsigned char* input, unsigned char* output, int cols, int rows) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int idx = y * cols + x;
        float gradientX = 0.0f, gradientY = 0.0f;

        // Calculate gradients (Sobel operators)
        if (x > 0 && x < cols - 1 && y > 0 && y < rows - 1) {
            gradientX = -1.0f * input[(y - 1) * cols + (x - 1)] + 1.0f * input[(y - 1) * cols + (x + 1)]
                - 2.0f * input[y * cols + (x - 1)] + 2.0f * input[y * cols + (x + 1)]
                - 1.0f * input[(y + 1) * cols + (x - 1)] + 1.0f * input[(y + 1) * cols + (x + 1)];

            gradientY = -1.0f * input[(y - 1) * cols + (x - 1)] - 2.0f * input[(y - 1) * cols + x] - 1.0f * input[(y - 1) * cols + (x + 1)]
                + 1.0f * input[(y + 1) * cols + (x - 1)] + 2.0f * input[(y + 1) * cols + x] + 1.0f * input[(y + 1) * cols + (x + 1)];
        }

        // Calculate gradient magnitude
        float gradientMagnitude = sqrtf(gradientX * gradientX + gradientY * gradientY);

        // Apply hysteresis thresholding to detect edges
        if (gradientMagnitude > 50) {  // Adjust this threshold as needed
            output[idx] = 255;
        }
        else {
            output[idx] = 0;
        }
    }
}

__global__ void gaussianBlurKernel(const unsigned char* input, unsigned char* output, int cols, int rows, int kernelSize, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int halfSize = kernelSize / 2;
        float sum = 0.0f;

        // Apply Gaussian blur using the kernel size
        for (int c = 0; c < channels; ++c) {
            sum = 0.0f;

            for (int i = -halfSize; i <= halfSize; ++i) {
                for (int j = -halfSize; j <= halfSize; ++j) {
                    int offsetX = x + i;
                    int offsetY = y + j;

                    if (offsetX >= 0 && offsetX < cols && offsetY >= 0 && offsetY < rows) {
                        float weight = exp(-(i * i + j * j) / (2.0f * kernelSize * kernelSize));
                        sum += weight * input[(offsetY * cols + offsetX) * channels + c];
                    }
                }
            }

            output[(y * cols + x) * channels + c] = static_cast<unsigned char>(sum);
        }
    }
}

__device__ void device_sort(unsigned char* values, int length) {
    // Sorting implementation using device-specific method
    // Example: bubble sort
    for (int i = 0; i < length - 1; ++i) {
        for (int j = 0; j < length - i - 1; ++j) {
            if (values[j] > values[j + 1]) {
                unsigned char temp = values[j];
                values[j] = values[j + 1];
                values[j + 1] = temp;
            }
        }
    }
}

__global__ void medianFilterKernel(const unsigned char* input, unsigned char* output, int cols, int rows, int channels, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int halfSize = kernelSize / 2;
        int kernelLength = kernelSize * kernelSize;

        // 필터 적용할 픽셀의 인덱스 계산
        int idx = (y * cols + x) * channels;

        // 중간값을 찾기 위한 임시 버퍼 생성
        unsigned char values[25]; // 최대 kernelSize는 5로 가정

        // 각 채널에 대해 median 필터 적용
        for (int c = 0; c < channels; ++c) {
            // 버퍼 초기화
            for (int i = 0; i < kernelLength; ++i) {
                int offsetX = x + (i % kernelSize) - halfSize;
                int offsetY = y + (i / kernelSize) - halfSize;

                // 경계 처리
                offsetX = max(0, min(cols - 1, offsetX));
                offsetY = max(0, min(rows - 1, offsetY));

                values[i] = input[(offsetY * cols + offsetX) * channels + c];
            }

            // 버퍼 정렬 후 중간값 취득 (직접 정렬 알고리즘 호출)
            device_sort(values, kernelLength);

            output[idx + c] = values[kernelLength / 2];
        }
    }
}

__global__ void laplacianFilterKernel(const unsigned char* input, unsigned char* output,
    int cols, int rows, size_t pitch, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        // Laplacian 필터 계산
        int sum = 0;
        sum += input[y * pitch + x * channels]; // 현재 픽셀

        if (x > 0)
            sum += input[y * pitch + (x - 1) * channels]; // 왼쪽 픽셀

        if (x < cols - 1)
            sum += input[y * pitch + (x + 1) * channels]; // 오른쪽 픽셀

        if (y > 0)
            sum += input[(y - 1) * pitch + x * channels]; // 위쪽 픽셀

        if (y < rows - 1)
            sum += input[(y + 1) * pitch + x * channels]; // 아래쪽 픽셀

        output[y * pitch + x * channels] = static_cast<unsigned char>(sum / 5); // Laplacian 필터 결과
    }
}

__global__ void bilateralKernel(const unsigned char* input, unsigned char* output, int width, int height, int kernelSize, int channels, float sigmaColor, float sigmaSpace) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half = kernelSize / 2;
    float colorCoeff = -0.5f / (sigmaColor * sigmaColor);
    float spaceCoeff = -0.5f / (sigmaSpace * sigmaSpace);

    for (int c = 0; c < channels; ++c) {
        float sum = 0;
        float norm = 0;

        for (int i = -half; i <= half; ++i) {
            for (int j = -half; j <= half; ++j) {
                int neighborX = min(max(x + j, 0), width - 1);
                int neighborY = min(max(y + i, 0), height - 1);

                int idx = (y * width + x) * channels + c;
                int nIdx = (neighborY * width + neighborX) * channels + c;

                float spaceDist = (i * i + j * j) * spaceCoeff;
                float colorDist = (input[idx] - input[nIdx]) * (input[idx] - input[nIdx]) * colorCoeff;

                float weight = expf(spaceDist + colorDist);
                sum += weight * input[nIdx];
                norm += weight;
            }
        }
        output[(y * width + x) * channels + c] = min(max(int(sum / norm), 0), 255);
    }
}

__global__ void sobelFilterKernel(const unsigned char* input, unsigned char* output,
    int cols, int rows, int channels) {
    // 스레드가 처리할 이미지의 픽셀 위치 계산
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        // 수직 및 수평 방향의 소벨 마스크 정의
        const int sobelX[3][3] = { {-1, 0, 1},
                                   {-2, 0, 2},
                                   {-1, 0, 1} };

        const int sobelY[3][3] = { {-1, -2, -1},
                                   {0, 0, 0},
                                   {1, 2, 1} };

        float gradX = 0.0f;
        float gradY = 0.0f;

        // 각 채널에 대해 소벨 필터 계산
        for (int c = 0; c < channels; ++c) {
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    int offsetX = x + j;
                    int offsetY = y + i;

                    if (offsetX >= 0 && offsetX < cols && offsetY >= 0 && offsetY < rows) {
                        int pixelIndex = (offsetY * cols + offsetX) * channels + c;
                        gradX += sobelX[i + 1][j + 1] * input[pixelIndex];
                        gradY += sobelY[i + 1][j + 1] * input[pixelIndex];
                    }
                }
            }
        }

        // 그래디언트 크기 계산 (마그니튜드)
        float magnitude = sqrtf(gradX * gradX + gradY * gradY);

        // 최종 그래디언트 값 (0-255 범위로 클리핑)
        for (int c = 0; c < channels; ++c) {
            output[(y * cols + x) * channels + c] = static_cast<unsigned char>(min(magnitude, 255.0f));
        }
    }
}


void callRotateImageCUDA(cv::Mat& inputImage, cv::Mat& outputImage) {
    int cols = inputImage.cols;
    int rows = inputImage.rows;
    int channels = inputImage.channels();

    uchar* d_inputImage = nullptr;
    uchar* d_outputImage = nullptr;
    size_t imageSize = cols * rows * channels * sizeof(uchar);

    cudaError_t err;
    err = cudaMalloc(&d_inputImage, imageSize);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_outputImage, imageSize);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        return;
    }

    err = cudaMemcpy(d_inputImage, inputImage.data, imageSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    rotateImageKernel << <numBlocks, threadsPerBlock >> > (d_inputImage, d_outputImage, cols, rows, channels);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return;
    }

    cudaDeviceSynchronize();

    outputImage.create(rows, cols, inputImage.type());

    err = cudaMemcpy(outputImage.data, d_outputImage, imageSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}

void callZoomImageCUDA(cv::Mat& inputImage, cv::Mat& outputImage, int newWidth, int newHeight) {
    int oldWidth = inputImage.cols;
    int oldHeight = inputImage.rows;
    int channels = inputImage.channels();

    uchar* d_inputImage = nullptr;
    uchar* d_outputImage = nullptr;
    size_t oldImageSize = oldWidth * oldHeight * channels * sizeof(uchar);
    size_t newImageSize = newWidth * newHeight * channels * sizeof(uchar);

    cudaError_t err;
    err = cudaMalloc(&d_inputImage, oldImageSize);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_outputImage, newImageSize);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        return;
    }

    err = cudaMemcpy(d_inputImage, inputImage.data, oldImageSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((newWidth + threadsPerBlock.x - 1) / threadsPerBlock.x, (newHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    resizeImageKernel << <numBlocks, threadsPerBlock >> > (d_inputImage, d_outputImage, oldWidth, oldHeight, newWidth, newHeight, channels);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return;
    }

    cudaDeviceSynchronize();

    outputImage.create(newHeight, newWidth, inputImage.type());

    err = cudaMemcpy(outputImage.data, d_outputImage, newImageSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}

void callGrayScaleImageCUDA(cv::Mat& inputImage, cv::Mat& outputImage) {
    int cols = inputImage.cols;
    int rows = inputImage.rows;
    int channels = inputImage.channels();

    if (channels != 3) {
        std::cerr << "Input image must be a 3-channel BGR image." << std::endl;
        return;
    }

    uchar* d_inputImage = nullptr;
    uchar* d_outputImage = nullptr;
    size_t inputSize = cols * rows * channels * sizeof(uchar);
    size_t outputSize = cols * rows * sizeof(uchar);

    cudaError_t err;
    err = cudaMalloc(&d_inputImage, inputSize);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_outputImage, outputSize);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        return;
    }

    err = cudaMemcpy(d_inputImage, inputImage.data, inputSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    grayScaleImageKernel << <numBlocks, threadsPerBlock >> > (d_inputImage, d_outputImage, cols, rows);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return;
    }

    cudaDeviceSynchronize();

    outputImage.create(rows, cols, CV_8UC1);
    err = cudaMemcpy(outputImage.data, d_outputImage, outputSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}

void callCannyEdgesCUDA(cv::Mat& inputImage, cv::Mat& outputImage) {
    int cols = inputImage.cols;
    int rows = inputImage.rows;
    int channels = inputImage.channels();

    //if (channels != 3) {
    //    std::cerr << "Input image must be a 3-channel BGR image." << std::endl;
    //    return;
    //}

    uchar* d_inputImage = nullptr;
    uchar* d_outputImage = nullptr;
    size_t inputSize = cols * rows * channels * sizeof(uchar);
    size_t outputSize = cols * rows * sizeof(uchar);

    cudaError_t err;
    err = cudaMalloc(&d_inputImage, inputSize);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_outputImage, outputSize);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        return;
    }

    err = cudaMemcpy(d_inputImage, inputImage.data, inputSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cannyEdgesKernel << <numBlocks, threadsPerBlock >> > (d_inputImage, d_outputImage, cols, rows);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return;
    }

    cudaDeviceSynchronize();

    outputImage.create(rows, cols, CV_8UC1);
    err = cudaMemcpy(outputImage.data, d_outputImage, outputSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}

void callGaussianBlurCUDA(cv::Mat& inputImage, cv::Mat& outputImage, int kernelSize) {
    int cols = inputImage.cols;
    int rows = inputImage.rows;
    int channels = inputImage.channels();

    uchar* d_inputImage = nullptr;
    uchar* d_outputImage = nullptr;
    size_t inputSize = cols * rows * channels * sizeof(uchar);
    size_t outputSize = cols * rows * channels * sizeof(uchar);

    cudaError_t err;
    err = cudaMalloc(&d_inputImage, inputSize);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_outputImage, outputSize);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        return;
    }

    err = cudaMemcpy(d_inputImage, inputImage.data, inputSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    gaussianBlurKernel << <numBlocks, threadsPerBlock >> > (d_inputImage, d_outputImage, cols, rows, kernelSize, channels);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return;
    }

    cudaDeviceSynchronize();

    outputImage.create(rows, cols, inputImage.type());
    err = cudaMemcpy(outputImage.data, d_outputImage, outputSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}

void callMedianFilterCUDA(cv::Mat & inputImage, cv::Mat& outputImage)
{
    // 이미지의 너비, 높이, 채널 수 확인
    int cols = inputImage.cols;
    int rows = inputImage.rows;
    int channels = inputImage.channels();

    // GPU 메모리 할당
    uchar* d_inputImage = nullptr;
    uchar* d_outputImage = nullptr;
    size_t inputSize = cols * rows * channels * sizeof(uchar);
    size_t outputSize = cols * rows * channels * sizeof(uchar);

    cudaError_t err;

    err = cudaMalloc(&d_inputImage, inputSize);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_outputImage, outputSize);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        return;
    }

    // CPU에서 GPU로 이미지 데이터 복사
    err = cudaMemcpy(d_inputImage, inputImage.data, inputSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy H2D error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return;
    }

    // CUDA 스레드 구성 설정
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 채널 수에 따라 적절한 커널 선택
    if (channels == 1 || channels == 3) {
        medianFilterKernel << <numBlocks, threadsPerBlock >> > (d_inputImage, d_outputImage, cols, rows, channels, 5);
    }
    else {
        std::cerr << "Unsupported number of channels: " << channels << std::endl;
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return;
    }

    // CUDA 커널 실행 오류 확인
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return;
    }

    // GPU에서 처리 완료 대기
    cudaDeviceSynchronize();

    // GPU에서 CPU로 결과 이미지 복사
    cv::Mat gpuOutputImage(rows, cols, inputImage.type());
    err = cudaMemcpy(gpuOutputImage.data, d_outputImage, outputSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy D2H error: " << cudaGetErrorString(err) << std::endl;
    }
    //else {
    //    inputImage = outputImage.clone();
    //}
    outputImage = gpuOutputImage.clone();

    // 메모리 해제
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}

void callLaplacianFilterCUDA(cv::Mat& inputImage) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();

    unsigned char* d_input;
    unsigned char* d_output;
    size_t pitch;

    cudaMallocPitch(&d_input, &pitch, width * channels * sizeof(unsigned char), height);
    cudaMallocPitch(&d_output, &pitch, width * channels * sizeof(unsigned char), height);

    cudaMemcpy2D(d_input, pitch, inputImage.ptr(), width * channels * sizeof(unsigned char), width * channels * sizeof(unsigned char), height, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    laplacianFilterKernel << <gridSize, blockSize >> > (d_input, d_output, width, height, pitch, channels);

    cudaMemcpy2D(inputImage.ptr(), width * channels * sizeof(unsigned char), d_output, pitch, width * channels * sizeof(unsigned char), height, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

void callBilateralFilterCUDA(cv::Mat& inputImage, int kernelSize, float sigmaColor, float sigmaSpace) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();

    cv::Mat outputImage(height, width, inputImage.type());

    unsigned char* d_input;
    unsigned char* d_output;
    size_t pitch;

    cudaMallocPitch(&d_input, &pitch, width * channels * sizeof(unsigned char), height);
    cudaMallocPitch(&d_output, &pitch, width * channels * sizeof(unsigned char), height);

    cudaMemcpy2D(d_input, pitch, inputImage.ptr(), width * channels * sizeof(unsigned char), width * channels * sizeof(unsigned char), height, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    bilateralKernel << <gridSize, blockSize >> > (d_input, d_output, width, height, kernelSize, channels, sigmaColor, sigmaSpace);

    cudaMemcpy2D(inputImage.ptr(), width * channels * sizeof(unsigned char), d_output, pitch, width * channels * sizeof(unsigned char), height, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

void callSobelFilterCUDA(cv::Mat& inputImage) {
    // 입력 이미지의 너비, 높이, 채널 수
    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();

    // CUDA 메모리 할당 및 복사
    unsigned char* d_input, * d_output;
    size_t pitch;
    cudaMallocPitch(&d_input, &pitch, width * channels * sizeof(unsigned char), height);
    cudaMallocPitch(&d_output, &pitch, width * channels * sizeof(unsigned char), height);

    cudaMemcpy2D(d_input, pitch, inputImage.ptr(), width * channels * sizeof(unsigned char), width * channels * sizeof(unsigned char), height, cudaMemcpyHostToDevice);

    // CUDA 블록 및 그리드 설정
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // CUDA 커널 호출
    sobelFilterKernel << <gridSize, blockSize >> > (d_input, d_output, width, height, channels);

    // CUDA에서 처리된 결과를 호스트로 복사
    cudaMemcpy2D(inputImage.ptr(), width * channels * sizeof(unsigned char), d_output, pitch, width * channels * sizeof(unsigned char), height, cudaMemcpyDeviceToHost);

    // 메모리 해제
    cudaFree(d_input);
    cudaFree(d_output);
}