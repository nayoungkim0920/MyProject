//imageProecssing.cu
#include "imageProcessing.cuh"

#define CUDA_CHECK_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        return; \
    }

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
        output[idx] = 0.299f * r + 0.587f * g + 0.114f * b;  // �׷��̽����� ��ȯ
    }
}

__global__ void cannyEdgesKernel(const unsigned char* input, unsigned char* output, int cols, int rows, int channels, bool isColor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int idx = y * cols + x;
        float gradientX = 0.0f, gradientY = 0.0f;

        // Calculate gradients (Sobel operators)
        if (x > 0 && x < cols - 1 && y > 0 && y < rows - 1) {
            for (int c = 0; c < (isColor ? 3 : 1); c++) {
                int offset = c * rows * cols;
                gradientX += -1.0f * input[offset + (y - 1) * cols + (x - 1)] + 1.0f * input[offset + (y - 1) * cols + (x + 1)]
                    - 2.0f * input[offset + y * cols + (x - 1)] + 2.0f * input[offset + y * cols + (x + 1)]
                    - 1.0f * input[offset + (y + 1) * cols + (x - 1)] + 1.0f * input[offset + (y + 1) * cols + (x + 1)];

                gradientY += -1.0f * input[offset + (y - 1) * cols + (x - 1)] - 2.0f * input[offset + (y - 1) * cols + x] - 1.0f * input[offset + (y - 1) * cols + (x + 1)]
                    + 1.0f * input[offset + (y + 1) * cols + (x - 1)] + 2.0f * input[offset + (y + 1) * cols + x] + 1.0f * input[offset + (y + 1) * cols + (x + 1)];
            }
        }

        // Calculate gradient magnitude
        float gradientMagnitude = sqrtf(gradientX * gradientX + gradientY * gradientY);

        // Apply hysteresis thresholding to detect edges
        if (gradientMagnitude > 50) {  // Adjust this threshold as needed
            if (isColor) {
                output[idx * 3] = 0;       // Blue
                output[idx * 3 + 1] = 255; // Green
                output[idx * 3 + 2] = 0;   // Red
            }
            else {
                output[idx] = 255;
            }
        }
        else {
            if (isColor) {
                output[idx * 3] = input[idx * 3];
                output[idx * 3 + 1] = input[idx * 3 + 1];
                output[idx * 3 + 2] = input[idx * 3 + 2];
            }
            else {
                output[idx] = 0;
            }
        }
    }
}

__global__ void gaussianBlurKernel(const unsigned char* input, unsigned char* output, int cols, int rows, int kernelSize, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int halfSize = kernelSize / 2;
        float sum = 0.0f;
        float normalization = 0.0f;

        for (int c = 0; c < channels; ++c) {
            sum = 0.0f;
            normalization = 0.0f;

            for (int i = -halfSize; i <= halfSize; ++i) {
                for (int j = -halfSize; j <= halfSize; ++j) {
                    int offsetX = x + i;
                    int offsetY = y + j;

                    if (offsetX >= 0 && offsetX < cols && offsetY >= 0 && offsetY < rows) {
                        float weight = expf(-(i * i + j * j) / (2.0f * halfSize * halfSize));
                        sum += weight * input[(offsetY * cols + offsetX) * channels + c];
                        normalization += weight;
                    }
                }
            }

            output[(y * cols + x) * channels + c] = static_cast<unsigned char>(sum / normalization);
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

        // ���� ������ �ȼ��� �ε��� ���
        int idx = (y * cols + x) * channels;

        // �߰����� ã�� ���� �ӽ� ���� ����
        unsigned char values[25]; // �ִ� kernelSize�� 5�� ����

        // �� ä�ο� ���� median ���� ����
        for (int c = 0; c < channels; ++c) {
            // ���� �ʱ�ȭ
            for (int i = 0; i < kernelLength; ++i) {
                int offsetX = x + (i % kernelSize) - halfSize;
                int offsetY = y + (i / kernelSize) - halfSize;

                // ��� ó��
                offsetX = max(0, min(cols - 1, offsetX));
                offsetY = max(0, min(rows - 1, offsetY));

                values[i] = input[(offsetY * cols + offsetX) * channels + c];
            }

            // ���� ���� �� �߰��� ��� (���� ���� �˰��� ȣ��)
            device_sort(values, kernelLength);

            output[idx + c] = values[kernelLength / 2];
        }
    }
}

__global__ void laplacianFilterKernel(const unsigned char* input, unsigned char* output,
    int cols, int rows, size_t pitch, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < cols - 1 && y >= 1 && y < rows - 1) {
        // Laplacian ���� ���
        int kernel[3][3] = {
            {0, -1, 0},
            {-1, 4, -1},
            {0, -1, 0}
        };

        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;

            // Ŀ�� ����
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int pixelX = x + kx;
                    int pixelY = y + ky;
                    float pixelValue = input[pixelY * pitch + pixelX * channels + c];
                    sum += pixelValue * kernel[ky + 1][kx + 1];
                }
            }

            // ��� ����
            output[y * pitch + x * channels + c] = static_cast<unsigned char>(min(max(sum, 0.0f), 255.0f));
        }
    }
}
__global__ void bilateralKernel(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int pitch, // pitch �߰�
    int kernelSize,
    int channels,
    float sigmaColor,
    float sigmaSpace
) {
    // ���� �ȼ� ��ǥ
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int radius = kernelSize / 2;
    float colorSum[3] = { 0.0f, 0.0f, 0.0f }; // �� ä���� ���� �ջ��� ���� �迭
    float weightSum = 0.0f;

    // ���� �ȼ� �� ��������
    unsigned char inputPixel[3];
    for (int c = 0; c < channels; ++c) {
        inputPixel[c] = d_input[y * pitch + x * channels + c]; // pitch ���
    }

    // Ŀ�� ��ȸ
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = min(max(x + dx, 0), width - 1);
            int ny = min(max(y + dy, 0), height - 1);

            unsigned char neighborPixel[3];
            for (int c = 0; c < channels; ++c) {
                neighborPixel[c] = d_input[ny * pitch + nx * channels + c]; // pitch ���
            }

            // ���� ���� �� ����ġ ���
            float colorDiff = 0.0f;
            for (int c = 0; c < channels; ++c) {
                float diff = inputPixel[c] - neighborPixel[c];
                colorDiff += diff * diff;
            }
            float colorWeight = expf(-colorDiff / (2.0f * sigmaColor * sigmaColor));

            float spatialDist = dx * dx + dy * dy;
            float spatialWeight = expf(-spatialDist / (2.0f * sigmaSpace * sigmaSpace));

            float weight = colorWeight * spatialWeight;
            weightSum += weight;

            for (int c = 0; c < channels; ++c) {
                colorSum[c] += neighborPixel[c] * weight;
            }
        }
    }

    // ��� �ȼ� �� ����
    for (int c = 0; c < channels; ++c) {
        d_output[y * pitch + x * channels + c] = static_cast<unsigned char>(min(max(colorSum[c] / weightSum, 0.0f), 255.0f));
    }
}

__global__ void sobelFilterKernel(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int sobelX[3][3] = { { -1, 0, 1 },
                         { -2, 0, 2 },
                         { -1, 0, 1 } };

    int sobelY[3][3] = { { -1, -2, -1 },
                         { 0, 0, 0 },
                         { 1, 2, 1 } };

    float gradientX = 0;
    float gradientY = 0;

    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);
            int pixelIdx = py * width + px;

            for (int c = 0; c < channels; ++c) {
                float pixelValue = d_input[pixelIdx * channels + c];
                gradientX += pixelValue * sobelX[ky + 1][kx + 1];
                gradientY += pixelValue * sobelY[ky + 1][kx + 1];
            }
        }
    }

    float magnitude = sqrtf(gradientX * gradientX + gradientY * gradientY);
    magnitude = min(max(magnitude, 0.0f), 255.0f); // Clip the values to be within 0-255

    if (channels == 1) {
        d_output[idx] = static_cast<unsigned char>(magnitude);
    }
    else if (channels == 3) {
        d_output[idx * 3 + 0] = static_cast<unsigned char>(magnitude); // Apply the same magnitude to all channels
        d_output[idx * 3 + 1] = static_cast<unsigned char>(magnitude);
        d_output[idx * 3 + 2] = static_cast<unsigned char>(magnitude);
    }
}

__global__ void rotateImageKernelR(const unsigned char* input, unsigned char* output, int cols, int rows, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        for (int c = 0; c < channels; ++c) {
            // ���������� 90�� ȸ��
            output[(x * rows + (rows - y - 1)) * channels + c] = input[(y * cols + x) * channels + c];
        }
    }
}

void callRotateImageCUDA_R(cv::Mat& inputImage, cv::Mat& outputImage) {
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

    rotateImageKernelR << <numBlocks, threadsPerBlock >> > (d_inputImage, d_outputImage, cols, rows, channels);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return;
    }

    cudaDeviceSynchronize();

    // ȸ���� �̹����� ũ�⸦ ����
    outputImage.create(cols, rows, inputImage.type());

    err = cudaMemcpy(outputImage.data, d_outputImage, imageSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}

__global__ void rotateImageKernelL(const unsigned char* input, unsigned char* output, int cols, int rows, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        for (int c = 0; c < channels; ++c) {
            // �������� 90�� ȸ��
            output[((cols - 1 - x) * rows + y) * channels + c] = input[(y * cols + x) * channels + c];
        }
    }
}

void callRotateImageCUDA_L(cv::Mat& inputImage, cv::Mat& outputImage) {
    int cols = inputImage.cols;
    int rows = inputImage.rows;
    int channels = inputImage.channels();

    uchar* d_inputImage = nullptr;
    uchar* d_outputImage = nullptr;
    size_t inputSize = cols * rows * channels * sizeof(uchar);
    size_t outputSize = rows * cols * channels * sizeof(uchar); // ȸ�� �� �̹��� ũ��

    cudaError_t err;
    err = cudaMalloc(&d_inputImage, inputSize);
    CUDA_CHECK_ERROR(err);

    err = cudaMalloc(&d_outputImage, outputSize);
    CUDA_CHECK_ERROR(err);

    err = cudaMemcpy(d_inputImage, inputImage.data, inputSize, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR(err);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    rotateImageKernelL << <numBlocks, threadsPerBlock >> > (d_inputImage, d_outputImage, cols, rows, channels);
    err = cudaGetLastError();
    CUDA_CHECK_ERROR(err);

    cudaDeviceSynchronize();

    // ȸ���� �̹����� ũ�⸦ ���� (�ʺ�� ���̰� �ٲ�)
    outputImage.create(cols, rows, inputImage.type());

    err = cudaMemcpy(outputImage.data, d_outputImage, outputSize, cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR(err);

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
    bool isColor = (channels == 3);

    uchar* d_inputImage = nullptr;
    uchar* d_outputImage = nullptr;
    size_t inputSize = cols * rows * channels * sizeof(uchar);
    size_t outputSize = cols * rows * (isColor ? channels : 1) * sizeof(uchar);

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

    cannyEdgesKernel << <numBlocks, threadsPerBlock >> > (d_inputImage, d_outputImage, cols, rows, channels, isColor);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return;
    }

    cudaDeviceSynchronize();

    if (isColor) {
        outputImage.create(rows, cols, CV_8UC3);
    }
    else {
        outputImage.create(rows, cols, CV_8UC1);
    }

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

    // CUDA �޸� �Ҵ�
    err = cudaMalloc(&d_inputImage, inputSize);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error for inputImage: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_outputImage, outputSize);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error for outputImage: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        return;
    }

    // CUDA �޸𸮷� ������ ����
    err = cudaMemcpy(d_inputImage, inputImage.data, inputSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error for inputImage: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return;
    }

    // CUDA Ŀ�� ȣ�� ����
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Gaussian blur Ŀ�� ȣ��
    gaussianBlurKernel << <numBlocks, threadsPerBlock >> > (d_inputImage, d_outputImage, cols, rows, kernelSize, channels);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return;
    }

    cudaDeviceSynchronize();

    // ��� �̹��� ���� �� ������ ����
    outputImage.create(rows, cols, inputImage.type());
    err = cudaMemcpy(outputImage.data, d_outputImage, outputSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error for outputImage: " << cudaGetErrorString(err) << std::endl;
    }

    // �޸� ����
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}

void callMedianFilterCUDA(cv::Mat & inputImage, cv::Mat& outputImage)
{
    // �̹����� �ʺ�, ����, ä�� �� Ȯ��
    int cols = inputImage.cols;
    int rows = inputImage.rows;
    int channels = inputImage.channels();

    // GPU �޸� �Ҵ�
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

    // CPU���� GPU�� �̹��� ������ ����
    err = cudaMemcpy(d_inputImage, inputImage.data, inputSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy H2D error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return;
    }

    // CUDA ������ ���� ����
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // ä�� ���� ���� ������ Ŀ�� ����
    if (channels == 1 || channels == 3) {
        medianFilterKernel << <numBlocks, threadsPerBlock >> > (d_inputImage, d_outputImage, cols, rows, channels, 5);
    }
    else {
        std::cerr << "Unsupported number of channels: " << channels << std::endl;
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return;
    }

    // CUDA Ŀ�� ���� ���� Ȯ��
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return;
    }

    // GPU���� ó�� �Ϸ� ���
    cudaDeviceSynchronize();

    // GPU���� CPU�� ��� �̹��� ����
    cv::Mat gpuOutputImage(rows, cols, inputImage.type());
    err = cudaMemcpy(gpuOutputImage.data, d_outputImage, outputSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy D2H error: " << cudaGetErrorString(err) << std::endl;
    }
    //else {
    //    inputImage = outputImage.clone();
    //}
    outputImage = gpuOutputImage.clone();

    // �޸� ����
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}

void callLaplacianFilterCUDA(cv::Mat& inputImage, cv::Mat& outputImage) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();

    unsigned char* d_input;
    unsigned char* d_output;
    size_t pitch;

    // CUDA �޸� �Ҵ�
    cudaMallocPitch(&d_input, &pitch, width * channels * sizeof(unsigned char), height);
    cudaMallocPitch(&d_output, &pitch, width * channels * sizeof(unsigned char), height);

    // �Է� �̹��� ����
    cudaMemcpy2D(d_input, pitch, inputImage.ptr(), inputImage.step[0], width * channels * sizeof(unsigned char), height, cudaMemcpyHostToDevice);

    // CUDA Ŀ�� ���� ����
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // CUDA Ŀ�� ȣ��
    laplacianFilterKernel << <gridSize, blockSize >> > (d_input, d_output, width, height, pitch, channels);

    // CUDA ���� üũ
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // ��� �̹��� ����
    outputImage.create(height, width, inputImage.type()); // �ùٸ� ���̿� �ʺ�� �̹��� ����
    cudaMemcpy2D(outputImage.ptr(), outputImage.step[0], d_output, pitch, width * channels * sizeof(unsigned char), height, cudaMemcpyDeviceToHost);

    // �޸� ����
    cudaFree(d_input);
    cudaFree(d_output);
}

void callBilateralFilterCUDA(
    cv::Mat& inputImage,
    cv::Mat& outputImage,
    int kernelSize,
    float sigmaColor,
    float sigmaSpace
) {
    if (inputImage.empty()) {
        std::cerr << "�Է� �̹����� ��� �ֽ��ϴ�." << std::endl;
        return;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();

    unsigned char* d_input;
    unsigned char* d_output;
    size_t pitch;

    // ��ġ �޸� �Ҵ� (pitch ���)
    cudaMallocPitch(&d_input, &pitch, width * channels * sizeof(unsigned char), height);
    cudaMallocPitch(&d_output, &pitch, width * channels * sizeof(unsigned char), height);

    // �Է� �̹����� ��ġ�� ����
    cudaMemcpy2D(d_input, pitch, inputImage.ptr(), inputImage.step, width * channels * sizeof(unsigned char), height, cudaMemcpyHostToDevice);

    // CUDA ��� �� �׸��� ũ�� ����
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Ŀ�� ����
    bilateralKernel << <gridSize, blockSize >> > (
        d_input, d_output, width, height, pitch, kernelSize, channels, sigmaColor, sigmaSpace
        );

    // ���� Ȯ��
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Ŀ�� ����: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // ��� �̹��� ����
    outputImage.create(height, width, inputImage.type());

    // ��ġ���� ȣ��Ʈ�� ��� �̹��� ����
    cudaMemcpy2D(outputImage.ptr(), outputImage.step, d_output, pitch, width * channels * sizeof(unsigned char), height, cudaMemcpyDeviceToHost);

    // ��ġ �޸� ����
    cudaFree(d_input);
    cudaFree(d_output);
}

void callSobelFilterCUDA(cv::Mat& inputImage, cv::Mat& outputImage) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();

    size_t pitch;
    unsigned char* d_input;
    unsigned char* d_output;

    // Allocate CUDA memory
    cudaMallocPitch(&d_input, &pitch, width * channels * sizeof(unsigned char), height);
    cudaMallocPitch(&d_output, &pitch, width * channels * sizeof(unsigned char), height);

    // Copy input image to device memory
    cudaMemcpy2D(d_input, pitch, inputImage.ptr(), inputImage.step[0], width * channels * sizeof(unsigned char), height, cudaMemcpyHostToDevice);

    // Define CUDA block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    if (channels == 1 || channels == 3) {
        // Apply Sobel filter
        sobelFilterKernel << <gridSize, blockSize >> > (d_input, d_output, width, height, channels);
        cudaDeviceSynchronize(); // Ensure kernel completion

        // Check for CUDA errors
        cudaError_t cudaErr = cudaGetLastError();
        if (cudaErr != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(cudaErr) << std::endl;
            cudaFree(d_input);
            cudaFree(d_output);
            return;
        }
    }
    else {
        std::cerr << "Unsupported number of channels: " << channels << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // Copy the result back to host memory
    outputImage.create(height, width, inputImage.type());
    cudaMemcpy2D(outputImage.ptr(), outputImage.step[0], d_output, pitch, width * channels * sizeof(unsigned char), height, cudaMemcpyDeviceToHost);

    // Free CUDA memory
    cudaFree(d_input);
    cudaFree(d_output);
}




void createGaussianKernel(float* kernel, int kernelSize, float sigma)
{
    int halfSize = kernelSize / 2;
    float sum = 0.0f;

    for (int i = -halfSize; i <= halfSize; ++i) {
        for (int j = -halfSize; j <= halfSize; ++j) {
            kernel[(i + halfSize) * kernelSize + (j + halfSize)] = expf(-(i * i + j * j) / (2.0f * sigma * sigma));
            sum += kernel[(i + halfSize) * kernelSize + (j + halfSize)];
        }
    }

    // Normalize the kernel
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            kernel[i * kernelSize + j] /= sum;
        }
    }
}
