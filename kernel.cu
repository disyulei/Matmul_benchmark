#include <cuda_runtime_api.h>
#include "kernel.hpp"



template <typename Dtype>
__global__ void matmul_naive(const Dtype* A, 
            const Dtype* B, Dtype* C,  
            const int row_A, const int col_A,
            const int col_B) {
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    size_t tid = xIndex + yIndex * col_B;
    if(xIndex >= col_B || yIndex >= row_A) return;

    Dtype tmp(0);
    for(int i = 0; i < col_A; ++i) {
        tmp += A[yIndex * col_A + i] * B[i*col_B + xIndex];
    }
    C[tid] = tmp;
}

#define DIM 32
#define IPAD 0

template <typename Dtype>
__global__ void matmul_shared(const Dtype* A, 
            const Dtype* B, Dtype* C,  
            const int row_A, const int col_A,
            const int col_B) {
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    size_t tid = xIndex + yIndex * col_B;
    if(xIndex >= col_B || yIndex >= row_A) return;

    Dtype tmp(0);
    for(int i = 0; i < col_A / DIM; ++i) {
        __shared__ Dtype sA[DIM][DIM+IPAD];
        __shared__ Dtype sB[DIM][DIM+IPAD];
        // load
        sA[threadIdx.y][threadIdx.x] = A[yIndex * col_A + DIM * i + threadIdx.x];
        sB[threadIdx.y][threadIdx.x] = B[xIndex + (DIM*i + threadIdx.y) * col_B];
        __syncthreads();
        // partial matmul
        for(int j = 0; j < DIM; ++j) {
            tmp += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
        __syncthreads();
    }
    if (col_A % DIM) { // remain 
        for(int i = col_A % DIM; i > 0; --i) tmp += A[yIndex * col_A + col_A - i] * B[xIndex + (col_A - i) * col_B];
    }

    C[tid] = tmp;
}




template <typename Dtype>
void matmul_naive_kernel(const Dtype* A, const Dtype* B, Dtype* C,
        const int m, const int n, const int k) {
    dim3 block(16, 16);
    int grid_x = (k + block.x - 1) / block.x;
    int grid_y = (m + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);

    matmul_naive<Dtype><<<grid, block>>>(
            A, B, C, m, n, k);
}

template <typename Dtype>
void matmul_shared_kernel(const Dtype* A, const Dtype* B, Dtype* C,
        const int m, const int n, const int k) {
    dim3 block(DIM, DIM);
    int grid_x = (k + block.x - 1) / block.x;
    int grid_y = (m + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);

    matmul_shared<Dtype><<<grid, block>>>(
            A, B, C, m, n, k);
}






template void matmul_naive_kernel(const float*, const float* ,float*, 
        const int, const int, const int );
template void matmul_naive_kernel(const double*, const double* , double*, 
        const int, const int, const int );
template void matmul_naive_kernel(const int*, const int* , int*, 
        const int, const int, const int );

template void matmul_shared_kernel(const float*, const float* ,float*, 
        const int, const int, const int );
template void matmul_shared_kernel(const double*, const double* , double*, 
        const int, const int, const int );
template void matmul_shared_kernel(const int*, const int* , int*, 
        const int, const int, const int );
