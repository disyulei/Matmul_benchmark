#include <gtest/gtest.h>
#include <vector>

#include "matmul.hpp"
#include <chrono>

using namespace std;
using namespace chrono;


TEST(Matrix, memory) {
    vector<float> a {1, 2, 3, 4};

    Matrix<float> A(a.data(), 2, 2);
    cout << "A: " << A << endl;

    Matrix<float> B(A);
    cout << " B: " << B << endl;

    Matrix<float> C = B;
    cout << " C: " << C << endl;

    Matrix<float> D(B, 0, 0, 1, 2);
    cout << " D: " << D << endl;

    Matrix<float> E(D);
    cout << " E: " << E << endl;

    Matrix<float> F = E;
    cout << " F: " << F << endl;

    Matrix<float>* A1 = new Matrix<float>(8, 8);
    *A1 = Matrix<float>::randn(8, 8);
    Matrix<float> B1(*A1, 1, 1, 3, 3);
    cout << " A:" << *A1 << endl;
    cout << " SUB:" << B1 << endl;

    Matrix<float> B2(B1, 1, 1, 2, 2);
    cout << " SUB: " << B2 << endl;
    delete A1;
}	


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

TEST(Matrix, copyFrom) {
    auto a = Matrix<float>::zeros(5, 5);
    auto b = Matrix<float>::randn(3, 3);
    a.copyFrom(b, 1, 1);
    cout << " A: " << a << "\t B:" << b << endl;
}

TEST(Matrix, matmul) {
    const int n = 4;
    auto A = Matrix<float>::randn(n, n);
    auto B = Matrix<float>::randn(n, n);

    auto C1 = Matrix<float>(n, n);
    auto C2 = Matrix<float>(n, n);
    matmul_naive(A, B, C1);
    matmul_openmp(A, B, C2);
    auto C4 = Matrix<float>(n, n);
    matmul_strassen(A, B, C4);

    auto C3 = Matrix<float>::zeros(n, n).cuda();

    matmul_cuda_naive(A.cuda(), B.cuda(), C3);
    cout << "A:" << A << "\nB:" << B << endl;
    cout << "A*B naive:"<< C1 << endl;
    cout << "A*B openmp:" << C2 << endl;
    cout << "A*B cuda:" << C3.cpu() << endl;
    cout << "strassen:" << C4 << endl;

    ASSERT_LE((C1-C4).abs().max(), 1e-5);
}


constexpr inline int U(const char* str)
{
	return str[0] + (str[1] ? U(str + 1) : 0);
}

class MatmulTest :  public testing::TestWithParam<tuple<std::string,int>> {
private:
    Matrix<float> _A;
    Matrix<float> _B;
    Matrix<float> _C;
    Matrix<float> _AB;
    string _name;
    int _n;
    high_resolution_clock::time_point _start;
    high_resolution_clock::time_point _end;
protected:
    void SetUp( ) {	
       _name = get<0>(GetParam());
       _n = get<1>(GetParam());

       _A = Matrix<float>::randn(_n, _n);
       _B = Matrix<float>::randn(_n, _n); 
       _C = Matrix<float>::zeros(_n, _n);
       _AB = Matrix<float>::zeros(_n, _n);
       matmul_naive(_A, _B, _AB);
       if (_name == "cuda-naive" || _name == "cuda-shared") {
          _A = _A.cuda(); _B = _B.cuda(); _C = _C.cuda();
       }
       _start = high_resolution_clock::now(); 
    }
    void TearDown( ) {
       switch(U(_name.c_str())) {
	  //case U("cpu-naive"): matmul_naive(_A, _B, _C); break;
	  case U("cpu-trans"): matmul_trans(_A, _B.transpose(), _C); break;
	  case U("cpu-omp"):   matmul_openmp(_A, _B, _C); break;
          case U("cpu-omp-sse"): matmul_omp_sse(_A, _B.transpose(), _C); break;
          case U("cpu-omp-avx"): matmul_omp_avx(_A, _B.transpose(), _C); break;			      
          case U("cpu-strassen"): matmul_strassen(_A, _B, _C); break;
	  case U("cuda-naive"): { matmul_cuda_naive(_A, _B, _C); cudaDeviceSynchronize(); } break;
	  case U("cuda-shared"): { matmul_cuda_shared(_A, _B, _C); cudaDeviceSynchronize(); } break;
          default: break;			      
       }
       _end = high_resolution_clock::now();
       if (_name == "cuda-naive" || _name == "cuda-shared") _C = _C.cpu();
       ASSERT_LE((_C - _AB).abs().max(), 5e-4);
       auto duration = std::chrono::duration<float, std::milli>(_end - _start).count();
       cout << "matmul --" << _name 
	    << " N:" << _n 
	    << " cost:" << duration 
	    << " ms" << endl;
    }	    
}; 	

TEST_P(MatmulTest, benchmark) {

}

//     ranges.begin(), ranges.end(), [] {static int i = -1; return pow(2, i++);


INSTANTIATE_TEST_CASE_P(
        CombineTest,
        MatmulTest,
        testing::Combine(testing::Values("cpu-trans", "cpu-omp", 
			"cpu-omp-sse", "cpu-omp-avx", //"cpu-strassen",
			"cuda-naive", "cuda-shared"),
			testing::Values(8, 16, 32, 64, 128, 256, 512, 1024)
			)
	);



int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    srand (time(NULL));
    return RUN_ALL_TESTS();
}
