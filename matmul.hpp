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
    bool _destroy {true} ;
    int _rows;
    int _cols;
    int _stride;
    Dtype* _data {nullptr};
    Dtype* _d_data {nullptr};
    Backend _backend; 
    int* _count; 

    void _free() {
       if (_destroy) {
           if(_data) { 
               //delete [] _data; 
	       free(_data);
               _data = nullptr;
           }
           if(_d_data) {
               CHECK_CUDA(cudaFree(_d_data));
               _d_data = nullptr;
           }
       }
       delete this->_count;
    }
public:
    Matrix() : _destroy(false), _rows(0), _cols(0), _stride(0), 
	_data(nullptr), _d_data(nullptr) {
        _count = new int(1);
    }
    Matrix(Dtype* data, const int rows, const int cols) { // deep copy 
        _rows = rows;
        _cols = cols;
	_stride = cols;
        _backend = Backend::CPU;
	_destroy = true;
        //_data = new Dtype[rows*cols];
        _data = static_cast<Dtype*>(aligned_alloc(32, rows*cols*sizeof(Dtype)));
	_d_data = nullptr;
        for(size_t i = 0; i < this->numel(); ++i) _data[i] = data[i]; 
        _count = new int(1);
    }
    Matrix(const int rows, const int cols) { // deep copy
        assert(rows > 0 && cols > 0);
        _rows = rows;
        _cols = cols;
	_stride = cols;
        _backend = Backend::CPU;
	_destroy = true;
        _data = static_cast<Dtype*>(aligned_alloc(32, rows*cols*sizeof(Dtype)));
	_d_data = nullptr;
        _count = new int(1);
    }
    Matrix(const Matrix<Dtype>& rhs) { // shadow copy
        _rows = rhs._rows;
        _cols = rhs._cols;
	_stride = rhs._stride;
        _backend = rhs._backend;
        _data = rhs._data;
        _d_data = rhs._d_data;
        _count = rhs._count;
        (*_count) ++;
	_destroy = rhs._destroy;
    }
    Matrix& operator=(const Matrix<Dtype>& rhs) { // shadow copy
        if (&rhs != this) {
            if(_d_data) {
                if (--(*_count) == 0) {
                    _free();
                }
            }
            _data = rhs._data;
            _d_data = rhs._d_data;
            _count = rhs._count;
            _backend = rhs._backend;
            _rows = rhs._rows;
            _cols = rhs._cols;
	    _stride = rhs._stride;
            (*_count) ++;
	    _destroy = rhs._destroy;
        }
        return *this;
    }
    Matrix(const Matrix<Dtype>& rhs, const int x, const int y, const int w, const int h) { //sub-matrix, not deep copy
	assert (rhs._data);
	assert (x >= 0 && x < rhs.cols() && y >= 0 && y < rhs.rows());
	assert (x+w <= rhs.cols() && y+h <= rhs.rows());
        _data = rhs._data + y * rhs.stride() + x;  	  
	_rows = h;
	_cols = w;
	_stride = rhs.stride();
	_backend = rhs._backend;
	_count = new int(1);
	_destroy = false; // never destroy _data
    }	    
    ~Matrix() {
        if (--(*_count) == 0) {
            _free();
        }
    }
    int rows() const {return _rows;}
    int cols() const {return _cols;}
    int stride() const {return _stride;}
    size_t numel() const {return _rows * _cols;}
    Dtype* data() { return _data;}
    const Dtype* data() const {return _data;}

    Dtype* operator[] (const int i) {
        return _data + i * _stride ;
    }
    const Dtype* operator[] (const int i) const {
        return _data + i * _stride;
    }
    Matrix<Dtype>& cuda() {
        assert (_data != nullptr && _destroy); 
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
        Matrix<Dtype> tmp(rows, cols);
        for(size_t i = 0; i < tmp.numel(); ++i) tmp.data()[i] = 0;
        return tmp;
    }
    static Matrix<Dtype> ones(const int rows, const int cols) {
        Matrix<Dtype> tmp (rows, cols);
        for(size_t i = 0; i < tmp.numel(); ++i) tmp.data()[i] = 1;
        return tmp;
    }
    static Matrix<Dtype> randn(const int rows, const int cols) {
        Matrix<Dtype> tmp (rows, cols);
        //const int seed = 0;
        size_t seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine gen(seed);
        std::normal_distribution<double> dis(0, 1);
        for(size_t i = 0; i < tmp.numel(); ++i) tmp.data()[i] = static_cast<Dtype>(dis(gen));

        return tmp;
    }


    void copyFrom(const Matrix<Dtype>& rhs, const int x, const int y) {
	assert (x+rhs.cols() <= _cols && y+rhs.rows() <= _rows);
	assert (_data);
	int i, j;
	#pragma omp parallel for private(i, j)
	for(i = 0; i < rhs.rows(); ++i) {
            for(j = 0; j < rhs.cols(); ++j) {
                (*this)[i+y][j+x] = rhs[i][j];
	    }
	}
    }

    Matrix<Dtype> operator-(const Matrix<Dtype>& rhs) {
        assert (this->_rows == rhs.rows() && this->_cols == rhs.cols());
        Matrix<Dtype> dst(rhs.rows(), rhs.cols());
	int i, j;
        #pragma omp parallel for private(i, j)
        for(i = 0; i < rhs.rows(); ++i) {
	    for(j = 0; j < rhs.cols(); ++j) {
            	dst[i][j] = (*this)[i][j] - rhs[i][j];
	    }
        }
        return dst;
    }
    Matrix<Dtype> operator+(const Matrix<Dtype>& rhs) {
        assert (this->_rows == rhs.rows() && this->_cols == rhs.cols());
        Matrix<Dtype> dst(rhs.rows(), rhs.cols());
	int i, j;
        #pragma omp parallel for private(i, j)
        for(i = 0; i < rhs.rows(); ++i) {
	    for(j = 0; j < rhs.cols(); ++j) {
            	dst[i][j] = (*this)[i][j] + rhs[i][j];
	    }
        }
        return dst;
    }
    Matrix<Dtype>& operator*=(const Dtype rhs) {
	int i, j;
        #pragma omp parallel for private(i, j)
        for(i = 0; i < _rows; ++i) {
	    for(j = 0; j < _cols; ++j) {
            	(*this)[i][j] *= rhs;
	    }
        }
        return *this;
    }
    Matrix<Dtype> abs() const {
        Matrix<Dtype> dst(_rows, _cols);
	int i, j;
        #pragma omp parallel for private(i, j)
	for(i = 0; i < _rows; ++i) {
            for(j = 0; j < _cols; ++j) {
                Dtype tmp = (*this)[i][j];
		dst[i][j] = tmp > 0 ? tmp: -tmp;
            }
        }
	return dst;
    }
    Dtype sum() const {
        Dtype dst(0);
	for(int i = 0; i < _rows; ++i) {
            for(int j = 0; j < _cols; ++j) {
                dst += (*this)[i][j];
	    }
	}
        return dst;
    }
    Dtype max() const {
        Dtype dst = (*this)[0][0];
	for(int i = 0; i < _rows; ++i) {
            for(int j = 0; j < _cols; ++j) {
                if(dst < (*this)[i][j]) dst = (*this)[i][j];
            }
        }		
	return dst;
    }

    Matrix<Dtype> transpose() {
        Matrix<Dtype> dst(_cols, _rows);
	int i, j;
        #pragma omp parallel for private(i, j)
        for(i = 0; i < dst.rows(); ++i) {
            for(j = 0; j < dst.cols(); ++j) {
                dst[i][j] = (*this)[j][i];
            }
        }
        return dst;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix<Dtype> & m) {
        os << "\n[";
        for(int i = 0; i < m._rows; ++i) {
            os << "[";
            for (int j = 0; j < m._cols; ++j) {
                os << m[i][j];
                if (j != m._cols - 1) os << ", ";
            }
            os << "]";
            if (i != m._rows - 1) os << ",\n";
        }
        os << "]";
        os << " device: " << m.device() ;
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
void matmul_trans(const Matrix<Dtype>& A, const Matrix<Dtype>& Bt, Matrix<Dtype>& C);

template <typename Dtype>
void matmul_trans_block(const Matrix<Dtype>& A, const Matrix<Dtype>& Bt, Matrix<Dtype>& C);


template <typename Dtype>
void matmul_omp_sse(const Matrix<Dtype>& A, const Matrix<Dtype>& Bt, Matrix<Dtype>& C);

template <typename Dtype>
void matmul_omp_avx(const Matrix<Dtype>& A, const Matrix<Dtype>& Bt, Matrix<Dtype>& C);
template <typename Dtype>
void matmul_omp_avx512(const Matrix<Dtype>& A, const Matrix<Dtype>& Bt, Matrix<Dtype>& C);

template <typename Dtype>
void matmul_strassen(const Matrix<Dtype>& A, const Matrix<Dtype>& B, Matrix<Dtype>& C);
