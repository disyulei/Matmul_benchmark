#include <gtest/gtest.h>
#include <vector>

#include "matmul.hpp"
#include <chrono>

using namespace std;
using namespace chrono;


TEST(Matrix, operator_index) {
    vector<int> a {1, 2, 3, 4};

    Matrix<int> A(a.data(), 2, 2);

    cout << " Matrix: " << A << endl;

    ASSERT_EQ(A[0][0], 1);
    ASSERT_EQ(A[0][1], 2);
    ASSERT_EQ(A[1][0], 3);
    ASSERT_EQ(A[1][1], 4);

    A[0][0] = 100;
    ASSERT_EQ(A[0][0], 100);
}

TEST(Matrix, zeros) {
    auto a = Matrix<int>::zeros(3, 2);
    cout << a << endl;

    auto b = Matrix<float>::zeros(3, 2);
    cout << b << endl;

    auto c(b);
    cout << c << endl;
}


TEST(Matrix, ones) {
    auto a = Matrix<int>::ones(3, 2);
    cout << a << endl;

    auto b = Matrix<float>::ones(3, 2);
    cout << b << endl;
}

TEST(Matrix, randn) {
    auto a = Matrix<float>::randn(5, 5);
    cout << "Random:" << a << endl;
}

TEST(Matrix, submat) {
    auto a = Matrix<float>::randn(10, 10);
    Matrix<float> b(a, 2, 2, 2, 2);
    cout << "A:" << a << "\n sub from (2,2):" << b ;
    Matrix<float>  c(a, 3, 3, 3, 3);
    cout << "\nsub from (3,3):" << c << endl;
}


TEST(Matrix, matmul) {
    auto A = Matrix<float>::randn(5, 5);
    auto B = Matrix<float>::randn(5, 5);

    auto C1 = Matrix<float>(5, 5);
    auto C2 = Matrix<float>(5, 5);
    matmul_naive(A, B, C1);
    matmul_openmp(A, B, C2);

    auto C3 = Matrix<float>::zeros(5, 5).cuda();

    matmul_cuda_naive(A.cuda(), B.cuda(), C3);
    cout << "A:" << A << "\nB:" << B << endl;
    cout << "A*B naive:"<< C1 << endl;
    cout << "A*B openmp:" << C2 << endl;
    cout << "A*B cuda:" << C3.cpu() << endl;
}


TEST(Matrix, benchmark) {
    const int N = 1024; 
    auto A = Matrix<float>::randn(N, N);
    auto B = Matrix<float>::randn(N, N);
    auto C1 = Matrix<float>(A.rows(), B.cols());
    auto C2 = Matrix<float>(A.rows(), B.cols());

    auto start = high_resolution_clock::now();
    matmul_naive(A, B, C1);
    auto end   = high_resolution_clock::now();
    auto duration = std::chrono::duration<float, std::milli>(end - start).count();
    cout << "matmul naive cost:" << duration << " ms" << endl;

    start = high_resolution_clock::now();
    matmul_openmp(A, B, C2);
    end   = high_resolution_clock::now();
    duration = std::chrono::duration<float, std::milli>(end - start).count();
    cout << "matmul openmp cost:" << duration << " ms" << endl;

    ASSERT_LE((C1 - C2).sum(), 1e-5);
    auto C3 = Matrix<float>(A.rows(), B.cols()).cuda();
    A = A.cuda();
    B = B.cuda();
    start = high_resolution_clock::now();
    matmul_cuda_naive(A, B, C3);
    cudaDeviceSynchronize();
    end   = high_resolution_clock::now();
    duration = std::chrono::duration<float, std::milli>(end - start).count();
    cout << "matmul cuda_naive cost:" << duration << " ms" << endl;

    C3 = C3.cpu();
    // ASSERT_LE((C1 - C3).sum(), 1e-5);
    ASSERT_LE((C1 - C3).max(), 1e-5);

    auto C4 = Matrix<float>::zeros(N, N).cuda();
    start = high_resolution_clock::now();
    matmul_cuda_shared(A, B, C4);
    cudaDeviceSynchronize();
    end   = high_resolution_clock::now();
    duration = std::chrono::duration<float, std::milli>(end - start).count();
    cout << "matmul cuda_shared cost:" << duration << " ms" << endl;
    
    ASSERT_LE((C1 - C4.cpu()).max(), 1e-5);

    auto C5 = Matrix<float>::zeros(N, N);
    start = high_resolution_clock::now();
    matmul_trans(A.cpu(), B.cpu().transpose(), C5);
    end   = high_resolution_clock::now();
    duration = std::chrono::duration<float, std::milli>(end - start).count();
    cout << "matmul trans cost:" << duration << " ms" << endl;
    
    ASSERT_LE((C1 - C5).max(), 1e-5);

    auto C6 = Matrix<float>::zeros(N, N);
    start = high_resolution_clock::now();
    matmul_omp_sse(A.cpu(), B.cpu().transpose(), C6);
    end   = high_resolution_clock::now();
    duration = std::chrono::duration<float, std::milli>(end - start).count();
    cout << "matmul sse cost:" << duration << " ms" << endl;
    
    ASSERT_LE((C1 - C6).max(), 1e-5);

    C6 = Matrix<float>::zeros(N, N);
    start = high_resolution_clock::now();
    matmul_omp_avx(A.cpu(), B.cpu().transpose(), C6);
    end   = high_resolution_clock::now();
    duration = std::chrono::duration<float, std::milli>(end - start).count();
    cout << "matmul avx cost:" << duration << " ms" << endl;
    
    ASSERT_LE((C1 - C6).max(), 1e-5);

}






int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    srand (time(NULL));
    return RUN_ALL_TESTS();
}
