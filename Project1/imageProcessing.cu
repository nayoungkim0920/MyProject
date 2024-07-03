#include "imageProcessing.cuh"
#include <opencv2/core/cuda.hpp>

// 전역 함수 정의
__device__ bool rotatePixel(int x, int y, int cols, int rows, int channels, const uchar* input, uchar* output) {
    if (x < cols && y < rows) {
        for (int c = 0; c < channels; ++c) {
            output[(x * rows + (rows - 1 - y)) * channels + c] = input[(y * cols + x) * channels + c];
        }
        return true;
    }
    return false;
}

// CUDA 커널
__global__ void rotateImageCUDA(const uchar* input, uchar* output, int cols, int rows, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (rotatePixel(x, y, cols, rows, channels, input, output)) {
        // 처리할 내용이 있으면 추가
    }
}


void callRotateImageCUDA(cv::Mat& inputImage) {
    int cols = inputImage.cols;
    int rows = inputImage.rows;
    int channels = inputImage.channels();

    uchar* d_inputImage;
    uchar* d_outputImage;
    size_t imageSize = cols * rows * channels * sizeof(uchar);

    cudaError_t err;
    err = cudaMalloc(&d_inputImage, imageSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMalloc(&d_outputImage, imageSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        cudaFree(d_inputImage);
        return;
    }

    err = cudaMemcpy(d_inputImage, inputImage.data, imageSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy error: %s\n", cudaGetErrorString(err));
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    rotateImageCUDA << <numBlocks, threadsPerBlock >> > (d_inputImage, d_outputImage, cols, rows, channels);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);
        return;
    }

    err = cudaMemcpy(inputImage.data, d_outputImage, imageSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}
