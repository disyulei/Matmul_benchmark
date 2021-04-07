#include <benchmark/benchmark.h>
#include "matmul.hpp"

static void BM_cpu_naive(benchmark::State& state) {
	const int n = static_cast<int>(state.range(0));
	auto A = Matrix<float>::randn(n, n);
	auto B = Matrix<float>::randn(n, n);
	auto C = Matrix<float>::zeros(n, n);

	for(auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		matmul_naive(A, B, C);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds = 
			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}
	state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_cpu_naive)->Name("cpu-naive")
		->RangeMultiplier(2)->Range(8, 2048)
		->Complexity(benchmark::oN)
		->UseManualTime()
		->Unit(benchmark::kMicrosecond);

static void BM_cpu_trans(benchmark::State& state) {
	const int n = static_cast<int>(state.range(0));
	auto A = Matrix<float>::randn(n, n);
	auto B = Matrix<float>::randn(n, n);
	auto C = Matrix<float>::zeros(n, n);

	for(auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		matmul_trans(A, B.transpose(), C);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds = 
			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}
	state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_cpu_trans)->Name("cpu-trans")
		->RangeMultiplier(2)->Range(8, 2048)
		->Complexity(benchmark::oN)
		->UseManualTime()
		->Unit(benchmark::kMicrosecond);

static void BM_cpu_trans_block(benchmark::State& state) {
	const int n = static_cast<int>(state.range(0));
	auto A = Matrix<float>::randn(n, n);
	auto B = Matrix<float>::randn(n, n);
	auto C = Matrix<float>::zeros(n, n);

	for(auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		matmul_trans_block(A, B.transpose(), C);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds = 
			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}
	state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_cpu_trans_block)->Name("cpu-trans-block")
		->RangeMultiplier(2)->Range(8, 2048)
		->Complexity(benchmark::oN)
		->UseManualTime()
		->Unit(benchmark::kMicrosecond);

static void BM_cpu_openmp(benchmark::State& state) {
	const int n = static_cast<int>(state.range(0));
	auto A = Matrix<float>::randn(n, n);
	auto B = Matrix<float>::randn(n, n);
	auto C = Matrix<float>::zeros(n, n);

	for(auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		matmul_openmp(A, B, C);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds = 
			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}
	state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_cpu_openmp)->Name("cpu-openmp")
		->RangeMultiplier(2)->Range(8, 2048)
		->Complexity(benchmark::oN)
		->UseManualTime()
		->Unit(benchmark::kMicrosecond);

static void BM_cpu_omp_sse(benchmark::State& state) {
	const int n = static_cast<int>(state.range(0));
	auto A = Matrix<float>::randn(n, n);
	auto B = Matrix<float>::randn(n, n);
	auto C = Matrix<float>::zeros(n, n);

	for(auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		matmul_omp_sse(A, B.transpose(), C);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds = 
			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}
	state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_cpu_omp_sse)->Name("cpu-omp-sse")
		->RangeMultiplier(2)->Range(8, 4096)
		->Complexity(benchmark::oN)
		->UseManualTime()
		->Unit(benchmark::kMicrosecond);

static void BM_cpu_omp_avx(benchmark::State& state) {
	const int n = static_cast<int>(state.range(0));
	auto A = Matrix<float>::randn(n, n);
	auto B = Matrix<float>::randn(n, n);
	auto C = Matrix<float>::zeros(n, n);

	for(auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		matmul_omp_avx(A, B.transpose(), C);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds = 
			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}
	state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_cpu_omp_avx)->Name("cpu-omp-avx")
		->RangeMultiplier(2)->Range(8, 4096)
		->Complexity(benchmark::oN)
		->UseManualTime()
		->Unit(benchmark::kMicrosecond);

static void BM_cpu_omp_avx512(benchmark::State& state) {
	const int n = static_cast<int>(state.range(0));
	auto A = Matrix<float>::randn(n, n);
	auto B = Matrix<float>::randn(n, n);
	auto C = Matrix<float>::zeros(n, n);

	for(auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		matmul_omp_avx512(A, B.transpose(), C);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds = 
			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}
	state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_cpu_omp_avx512)->Name("cpu-omp-avx512")
		->RangeMultiplier(2)->Range(8, 4096)
		->Complexity(benchmark::oN)
		->UseManualTime()
		->Unit(benchmark::kMicrosecond);

static void BM_cuda_naive(benchmark::State& state) {
	const int n = static_cast<int>(state.range(0));
	auto A = Matrix<float>::randn(n, n);
	auto B = Matrix<float>::randn(n, n);
	auto C = Matrix<float>::zeros(n, n);
	 A = A.cuda(); B = B.cuda(); C = C.cuda();

	for(auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		matmul_cuda_naive(A, B, C);
		cudaDeviceSynchronize();
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds = 
			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}
	state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_cuda_naive)->Name("cuda-naive")
		->RangeMultiplier(2)->Range(8, 8192)
		->Complexity(benchmark::oN)
		->UseManualTime()
		->Unit(benchmark::kMicrosecond);

static void BM_cuda_shared(benchmark::State& state) {
	const int n = static_cast<int>(state.range(0));
	auto A = Matrix<float>::randn(n, n);
	auto B = Matrix<float>::randn(n, n);
	auto C = Matrix<float>::zeros(n, n);
	 A = A.cuda(); B = B.cuda(); C = C.cuda();

	for(auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		matmul_cuda_shared(A, B, C);
		cudaDeviceSynchronize();
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds = 
			std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}
	state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_cuda_shared)->Name("cuda-shared")
		->RangeMultiplier(2)->Range(8, 8192)
		->Complexity(benchmark::oN)
		->UseManualTime()
		->Unit(benchmark::kMicrosecond);


BENCHMARK_MAIN();
