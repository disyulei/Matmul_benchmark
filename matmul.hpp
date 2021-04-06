#pragma once

#include <iostream>
#include <cassert>
#include <random>
#include <chrono>
#include <cuda_runtime_api.h>

#include "kernel.hpp"


struct CPU {};
struct GPU {};
enum class Backend {
    CPU, 
    GPU
};
#define CHECK_CUDA(e) { if(e != cudaSuccess) { \
    printf("cuda failure: %s:%d: '%s'\n", __FILE__, __LINE__, \
            cudaGetErrorString(e)); \
        exit(0); \
    } \
}


template <typename Dtype>
class Matrix {
private:
    Dtype* _data {nullptr};
    Dtype* _d_data {nullptr};
    int _rows;
    int _cols;
    Backend _backend; 
public:
    Matrix(Dtype* data, const int rows, const int cols) {
        _rows = rows;
        _cols = cols;
        _backend = Backend::CPU;
        //_data = new Dtype[rows*cols];
	    _data = static_cast<Dtype*>(aligned_alloc(32, rows*cols*sizeof(Dtype))); // force the data aligned with 32 bytes, required by AVX 
        for(size_t i = 0; i < this->numel(); ++i) _data[i] = data[i]; 
    }
    Matrix(const int rows, const int cols) {
        assert(rows > 0 && cols > 0);
        this->_rows = rows;
        this->_cols = cols;
        //_data = new Dtype[rows*cols]; 
	    _data = static_cast<Dtype*>(aligned_alloc(32, rows*cols*sizeof(Dtype)));
        _backend = Backend::CPU;
    }
    Matrix()=delete; 
    Matrix(const Matrix<Dtype>& rhs) {
        _rows = rhs._rows;
        _cols = rhs._cols;
        _backend = rhs._backend;
        if (rhs.numel() > 0) {
            //_data = new Dtype[rhs.numel()];
	        _data = static_cast<Dtype*>(aligned_alloc(32, rhs.numel()*sizeof(Dtype)));

            for(size_t i = 0; i < rhs.numel(); ++i) _data[i] = rhs.data()[i]; 
            if (rhs._d_data && rhs._backend == Backend::GPU) {
                CHECK_CUDA(cudaMalloc((void**)&_d_data, sizeof(Dtype) * _rows * _cols));
                CHECK_CUDA(cudaMemcpy(_d_data, rhs._d_data, sizeof(Dtype) * _rows * _cols, cudaMemcpyDeviceToDevice));
            }
        }
    }
    Matrix& operator=(const Matrix<Dtype>& rhs) {
        if (&rhs != this) {
            _rows = rhs._rows;
            _cols = rhs._cols;
            _backend = rhs._backend;
            if (rhs.numel() > 0) {
                //_data = new Dtype[rhs.numel()];
	            _data = static_cast<Dtype*>(aligned_alloc(32, rhs.numel()*sizeof(Dtype)));
                for(size_t i = 0; i < rhs.numel(); ++i) _data[i] = rhs.data()[i]; 
                if (rhs._d_data && rhs._backend == Backend::GPU) {
                    CHECK_CUDA(cudaMalloc((void**)&_d_data, sizeof(Dtype) * _rows * _cols));
                    CHECK_CUDA(cudaMemcpy(_d_data, rhs._d_data, sizeof(Dtype) * _rows * _cols, cudaMemcpyDeviceToDevice));
                }
            }
        }
        return *this;
    }
    ~Matrix() {
        if(_data) { 
            delete [] _data; 
            _data = nullptr;
        }
        if(_d_data) {
            CHECK_CUDA(cudaFree(_d_data));
            _d_data = nullptr;
        }
    }
    int rows() const {return _rows;}
    int cols() const {return _cols;}
    size_t numel() const {return _rows * _cols;}
    Dtype* data() {return _data;}
    const Dtype* data() const {return _data;}

    Dtype* operator[] (const int i) {
        return _data + i * _cols ;
    }
    const Dtype* operator[] (const int i) const {
        return _data + i * _cols;
    }
    Matrix<Dtype>& cuda() {
        assert (_data != nullptr);
        if (_backend == Backend::CPU) {
            _backend = Backend::GPU;
            if (nullptr == _d_data) {
                CHECK_CUDA(cudaMalloc((void**)&_d_data, sizeof(Dtype) * _rows * _cols));
            }
            CHECK_CUDA(cudaMemcpy(_d_data, _data, sizeof(Dtype) * _rows * _cols, cudaMemcpyHostToDevice));
        }
        return *this;
    }
    Matrix<Dtype>& cpu() {
        if (_backend == Backend::GPU) {
            _backend = Backend::CPU;
            if (_d_data) CHECK_CUDA(cudaMemcpy(_data, _d_data, sizeof(Dtype) * _rows * _cols, cudaMemcpyDeviceToHost));
        }
        return *this;
    }
    const char* device() const {
        if (_backend == Backend::CPU) return "CPU";
        return "GPU";
    }
    void* gpu_data() const { return reinterpret_cast<void*>(_d_data); }

    static Matrix<Dtype> zeros(const int rows, const int cols) {
        Matrix<Dtype>* tmp = new Matrix<Dtype>(rows, cols);
        for(size_t i = 0; i < rows*cols; ++i) tmp->data()[i] = 0;
        return *tmp;
    }
    static Matrix<Dtype> ones(const int rows, const int cols) {
        Matrix<Dtype>* tmp = new Matrix<Dtype>(rows, cols);
        for(size_t i = 0; i < rows*cols; ++i) tmp->data()[i] = 1;
        return *tmp;
    }
    static Matrix<Dtype> randn(const int rows, const int cols) {
        auto tmp = new Matrix<Dtype>(rows, cols);
        //const int seed = 0;
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine gen(seed);
        std::normal_distribution<double> dis(0,1);
        for(size_t i = 0; i < rows*cols; ++i) tmp->data()[i] = static_cast<Dtype>(dis(gen));

        return *tmp;
    }

    Matrix<Dtype> operator-(const Matrix<Dtype>& rhs) {
        assert (this->_rows == rhs.rows() && this->_cols == rhs.cols());
        Matrix<Dtype> dst(rhs.rows(), rhs.cols());
        for(size_t i = 0; i < rhs.rows()*rhs.cols(); ++i) {
            dst.data()[i] = this->_data[i] - rhs.data()[i];
        }
        return dst;
    }

    Dtype sum() const {
        Dtype dst(0);
        for(size_t i = 0; i < _rows * _cols; ++i) dst += _data[i];
        return dst;
    }
    Dtype max() const {
        Dtype dst(0);
        for(size_t i = 0; i < this->numel(); ++i) {
            if (dst < _data[i]) dst = _data[i];
        }
    }

    Matrix<Dtype> transpose() {
        Matrix<Dtype> dst(_cols, _rows);
        for(int i = 0; i < _rows; ++i) {
            for(int j = 0; j < _cols; ++j) {
                dst[j][i] = _data[i * _rows + j];
            }
        }
        return dst;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix<Dtype> & m) {
        os << "[";
        for(int i = 0; i < m._rows; ++i) {
            os << "[";
            for (int j = 0; j < m._cols; ++j) {
                os << m[i][j];
                if (j != m._cols - 1) os << ", ";
            }
            os << "]";
            if (i != m._rows - 1) os << "\n";
        }
        os << "]";
        os << " " << m.device() ;
        return os;
    }

};



template <typename Dtype>
void matmul_naive(const Matrix<Dtype>& A, const Matrix<Dtype>& B, Matrix<Dtype>& C);

template <typename Dtype>
void matmul_openmp(const Matrix<Dtype>& A, const Matrix<Dtype>& B, Matrix<Dtype>& C);

template <typename Dtype>
void matmul_cuda_naive(const Matrix<Dtype>& A, const Matrix<Dtype>& B, Matrix<Dtype>& C);

template <typename Dtype>
void matmul_cuda_shared(const Matrix<Dtype>& A, const Matrix<Dtype>& B, Matrix<Dtype>& C);

template <typename Dtype>
void matmul_trans(const Matrix<Dtype>& A, const Matrix<Dtype>& B, Matrix<Dtype>& C);


template <typename Dtype>
void matmul_omp_sse(const Matrix<Dtype>& A, const Matrix<Dtype>& Bt, Matrix<Dtype>& C);

template <typename Dtype>
void matmul_omp_avx(const Matrix<Dtype>& A, const Matrix<Dtype>& Bt, Matrix<Dtype>& C);

