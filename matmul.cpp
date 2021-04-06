#include <iostream>
#include <xmmintrin.h> // SSE
#include <immintrin.h> // AVX

#include "matmul.hpp"


template <typename Dtype>
void matmul_naive(const Matrix<Dtype>& A, const Matrix<Dtype>& B, Matrix<Dtype>& C) {
    assert (A.rows() == C.rows() && A.cols() == B.rows() && B.cols() == C.cols());
    for(int i = 0; i < C.rows(); ++i) {
        for(int j = 0; j < C.cols(); ++j) {
            Dtype tmp(0);
            for(int k = 0; k < A.cols(); ++k) {
                tmp += A[i][k] * B[k][j];
            }
            C[i][j] = tmp;
        }
    }
}


template <typename Dtype>
void matmul_openmp(const Matrix<Dtype>& A, const Matrix<Dtype>& B, Matrix<Dtype>& C) {
    assert (A.rows() == C.rows() && A.cols() == B.rows() && B.cols() == C.cols());
    int i, j, k;
    #pragma omp parallel for private(i, j)
    for(i = 0; i < C.rows(); ++i) {
        for(j = 0; j < C.cols(); ++j) {
            Dtype tmp(0);
            for(k = 0; k < A.cols(); ++k) {
                tmp += A[i][k] * B[k][j];
            }
            C[i][j] = tmp;
        }
    }
}


template <typename Dtype>
void matmul_trans(const Matrix<Dtype>& A, const Matrix<Dtype>& B, Matrix<Dtype>& C) {
    assert (A.rows() == C.rows() && A.cols() == B.rows() && B.cols() == C.cols());
    int i, j;
    for(i = 0; i < C.rows(); ++i) {
        for(j = 0; j < C.cols(); ++j) {
            Dtype tmp(0);
            for(int k = 0; k < A.cols(); ++k) {
                tmp += A[i][k] * B[j][k];
            }
            C[i][j] = tmp;
        }
    }
}





template <typename Dtype>
void matmul_cuda_naive(const Matrix<Dtype>& A, const Matrix<Dtype>& B, Matrix<Dtype>& C) {
    matmul_naive_kernel<Dtype>(
            reinterpret_cast<Dtype*>(A.gpu_data()), 
            reinterpret_cast<Dtype*>(B.gpu_data()), 
            reinterpret_cast<Dtype*>(C.gpu_data()), 
            A.rows(), A.cols(), B.cols());
}

template <typename Dtype>
void matmul_cuda_shared(const Matrix<Dtype>& A, const Matrix<Dtype>& B, Matrix<Dtype>& C) {
    matmul_shared_kernel<Dtype>(
            reinterpret_cast<Dtype*>(A.gpu_data()), 
            reinterpret_cast<Dtype*>(B.gpu_data()), 
            reinterpret_cast<Dtype*>(C.gpu_data()), 
            A.rows(), A.cols(), B.cols());
}



template <>
void matmul_omp_sse(const Matrix<float>& A, const Matrix<float>& Bt, Matrix<float>& C) {
    // data must be aligned with 16 bytes
    int i, j;
    #pragma omp parallel for private(i, j)
    for(i = 0; i < C.rows(); ++i) {
	    for(j = 0; j < C.cols(); ++j) {
	        __m128 partial_sum = _mm_setzero_ps();
	        int k ; 
            for(k = 0; k < A.cols() / 4; k += 4) {
                auto a = _mm_load_ps(A.data() + i * A.cols() + 4 * k);
                auto b = _mm_load_ps(Bt.data() + i * Bt.cols() + 4 * k);
                auto c = _mm_mul_ps(a, b); 
	            partial_sum = _mm_add_ps(partial_sum, c);	
            }
	        __attribute__ ((aligned (16))) float addr[4];
	        _mm_store_ps(addr, partial_sum);

	        C[i][j] = addr[0] + addr[1] + addr[2] + addr[3];
	    }
    }
}

template <>
void matmul_omp_avx(const Matrix<float>& A, const Matrix<float>& Bt, Matrix<float>& C) {
    // data must be aligned with 32 bytes
    int i, j;
    #pragma omp parallel for private(i, j)
    for(i = 0; i < C.rows(); ++i) {
	    for(j = 0; j < C.cols(); ++j) {
	        auto partial_sum = _mm256_setzero_ps();
	        int k ; 
            for(k = 0; k < A.cols() / 8; k += 8) {
                auto a = _mm256_load_ps(A.data() + i * A.cols() + 8 * k);
                auto b = _mm256_load_ps(Bt.data() + i * Bt.cols() + 8 * k);
                auto c = _mm256_mul_ps(a, b); 
	            partial_sum = _mm256_add_ps(partial_sum, c);	
            }
	        __attribute__ ((aligned (32))) float addr[8];
	        _mm256_store_ps(addr, partial_sum);

	        C[i][j] = addr[0] + addr[1] + addr[2] + addr[3] +
	    	    addr[4] + addr[5] + addr[6] + addr[7];
	    }
    }
}



template void matmul_naive(const Matrix<float>&, const Matrix<float>&, Matrix<float>& );
template void matmul_naive(const Matrix<double>&, const Matrix<double>&, Matrix<double>& );
template void matmul_naive(const Matrix<int>&, const Matrix<int>&, Matrix<int>& );

template void matmul_openmp(const Matrix<float>&, const Matrix<float>&, Matrix<float>& );
template void matmul_openmp(const Matrix<double>&, const Matrix<double>&, Matrix<double>& );
template void matmul_openmp(const Matrix<int>&, const Matrix<int>&, Matrix<int>& );

template void matmul_cuda_naive(const Matrix<float>&, const Matrix<float>&, Matrix<float>& );
template void matmul_cuda_naive(const Matrix<double>&, const Matrix<double>&, Matrix<double>& );
template void matmul_cuda_naive(const Matrix<int>&, const Matrix<int>&, Matrix<int>& );

template void matmul_cuda_shared(const Matrix<float>&, const Matrix<float>&, Matrix<float>& );
template void matmul_cuda_shared(const Matrix<double>&, const Matrix<double>&, Matrix<double>& );
template void matmul_cuda_shared(const Matrix<int>&, const Matrix<int>&, Matrix<int>& );

template void matmul_trans(const Matrix<float>&, const Matrix<float>&, Matrix<float>& );
template void matmul_trans(const Matrix<double>&, const Matrix<double>&, Matrix<double>& );
template void matmul_trans(const Matrix<int>&, const Matrix<int>&, Matrix<int>& );


