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

            for(int k = 0; k < A.cols() / 4; ++k) {
                auto a = _mm_load_ps(A.data() + i * A.cols() + 4 * k);
                auto b = _mm_load_ps(Bt.data() + j * Bt.cols() + 4 * k);
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
            for(k = 0; k < A.cols() / 8; ++k) {
                auto a = _mm256_load_ps(A.data() + i * A.cols() + 8 * k);
                auto b = _mm256_load_ps(Bt.data() + j * Bt.cols() + 8 * k);
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

template <typename Dtype>
__inline__ void _strassen_2x2(const Dtype* a, const int stride_a, 
		const Dtype* b, const int stride_b,
		Dtype* c, const int stride_c) {
    Dtype p1 = (b[1] - b[stride_b + 1]) * a[0];
    Dtype p2 = (a[0] + a[1]) * b[stride_b + 1];
    Dtype p3 = (a[stride_a] + a[stride_a+1]) * b[0];
    Dtype p4 = (b[stride_b] - b[0]) * a[stride_a+1];
    Dtype p5 = (a[0] + a[stride_a+1]) * (b[0] + b[stride_b+1]);
    Dtype p6 = (a[1] - a[stride_a+1]) * (b[stride_b] + b[stride_b+1]);
    Dtype p7 = (a[0] - a[stride_a]) * (b[0] + b[1]);

    c[0] = p5 + p4 - p2 + p6;
    c[1] = p1 + p2;
    c[stride_c] = p3 + p4;
    c[stride_c+1] = p1 + p5 - p3 - p7;
}


template <typename Dtype>
void matmul_strassen(const Matrix<Dtype>& A, 
		const Matrix<Dtype>& B,  
		Matrix<Dtype>& C) {
     assert (A.rows() == A.cols());
     const int n = A.rows();
     assert ((n & n-1) == 0);
     if (n == 2) {
	_strassen_2x2(A.data(), A.stride(), B.data(), B.stride(), C.data(), C.stride());
	//n=1, C.data()[0] = A.data()[0] * B.data()[0];
	return;
     }
     const int half_n = n >> 1;
     // split
     Matrix<Dtype> A11(A, 0, 0, half_n, half_n);  // sub-matrix
     Matrix<Dtype> A12(A, half_n, 0, half_n, half_n); 
     Matrix<Dtype> A21(A, 0, half_n, half_n, half_n);
     Matrix<Dtype> A22(A, half_n, half_n, half_n, half_n); 

     Matrix<Dtype> B11(B, 0, 0, half_n, half_n);
     Matrix<Dtype> B12(B, half_n, 0, half_n, half_n);
     Matrix<Dtype> B21(B, 0, half_n, half_n, half_n);
     Matrix<Dtype> B22(B, half_n, half_n, half_n, half_n);


     Matrix<Dtype> P1 = B12 - B22;
     Matrix<Dtype> P2 = A11 + A12;
     Matrix<Dtype> P3 = A21 + A22;
     Matrix<Dtype> P4 = B21 - B11;
     Matrix<Dtype> P51 = A11 + A22;
     Matrix<Dtype> P52 = B11 + B22;
     Matrix<Dtype> P61 = A12 - A22;
     Matrix<Dtype> P62 = B21 + B22;
     Matrix<Dtype> P71 = A11 - A21;
     Matrix<Dtype> P72 = B11 + B12;


     matmul_strassen(A11, P1, P1);
     matmul_strassen(P2, B22, P2);
     matmul_strassen(P3, B11, P3);
     matmul_strassen(A22, P4, P4);
     matmul_strassen(P51, P52, P52);
     matmul_strassen(P61, P62, P62);
     matmul_strassen(P71, P72, P72);


     // merge
     int i, j;
     #pragma omp parallel for private(i, j)
     for(i = 0; i < half_n; ++i) {
        for(j = 0; j < half_n; ++j) {
           C[i][j] = P52[i][j] + P4[i][j] - P2[i][j] + P62[i][j];
        }
     }       
     #pragma omp parallel for private(i, j)
     for(i = 0; i < half_n; ++i) {
        for(j = 0; j < half_n; ++j) {
           C[i][j+half_n] = P1[i][j] + P2[i][j]; 
        }
     }       
     #pragma omp parallel for private(i, j)
     for(i = 0; i < half_n; ++i) {
        for(j = 0; j < half_n; ++j) {
           C[i+half_n][j] = P3[i][j] + P4[i][j]; 
        }
     }       
     #pragma omp parallel for private(i, j)
     for(i = 0; i < half_n; ++i) {
        for(j = 0; j < half_n; ++j) {
           C[i+half_n][j+half_n] = P1[i][j] + P52[i][j] - P3[i][j] - P72[i][j]; 
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

template void matmul_strassen(const Matrix<float>&, const Matrix<float>&, Matrix<float>& );
template void matmul_strassen(const Matrix<double>&, const Matrix<double>&, Matrix<double>& );
template void matmul_strassen(const Matrix<int>&, const Matrix<int>&, Matrix<int>& );
