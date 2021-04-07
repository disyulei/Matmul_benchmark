# Matrix Multiplication Optimization Methods

This is a toy benchmark experiment of several matrix multiplication algorithms. 

## Requirements
- googletest
- google/benchmark
- openmp
- CUDA

## Algorithms
- Naive: the pure method for calculating `C = A * B`, the time complexity is `O(N^3)`
```
for(int i = 0; i < A.rows; ++i) {
    for(int j = 0; j < B.cols; ++j) {
        C[i][j] = 0;
        for(int k = 0; k < A.cols; ++k) {
            C[i][j] += A[i][k] * B[k][i];
        }
    }
}
```
- Naive-Trans:
in `Naive`, the access of `B[:][i]` is not cache-friendly, and we can transpose `B` (cost `O(N^2)`) before doing matmul, 
```
// Bt = B.transpose()
for(int i = 0; i < B.rows; ++i) {
    for(int j = 0; j < B.cols; ++j) {
        Bt[j][i] = B[i][j];
    }
}
// C = A * B
for(int i = 0; i < A.rows; ++i) {
    for(int j = 0; j < B.cols; ++j) {
        C[i][j] = 0;
        for(int k = 0; k < A.cols; ++k) {
            C[i][j] += A[i][k] * Bt[i][k]; // cache friendly 
        }
    }
}
```
- Using OpenMP : a simple method via multi-cores

- Using SIMD (CUDA) : naive method, shared memory based method

- Using SIMD (CPU): Intel SSE, AVX instructions, etc.

- Strassen (Divide and Conquer): 
1. naive
```
A = [a, b; c, d] B = [e, f; g, h]  C=A*B=[ae +bg, af+bh; ce+dg, cf+dh]
```
which needs 8 multiplications and 4 additions. the time complexity 
```

T(n) = 8T(n/2) + O(n^2)
T(n) = O(n^log2(8)) = O(n^3)
```
2. strassen 
```
A = [a, b; c, d] B = [e, f; g, h]  C=A*B
p1 = a(f-h), p2 = (a+b)h
p3 = (c+d)e, p4 = d(g-e)
p5 = (a+d)(e+h), p6 = (b-d)(g+h)
p7 = (a-c)(e+f)

C = [p5+p4-p2+p6, p1+p2; p3+p4, p1+p5-p3-p7]
```
```
T(n) = 7T(n/2) + O(n^2)
T(n) = O(n^log2(7)) ~ O(n^2.8074)
```


## Benchmark
run the benchmark via the executable `bench`, an example log : 
```
Running ./bench
Run on (48 X 3500 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x24)
  L1 Instruction 32 KiB (x24)
  L2 Unified 1024 KiB (x24)
  L3 Unified 16896 KiB (x2)
Load Average: 15.18, 15.77, 8.19
--------------------------------------------------------------------------
Benchmark                                Time             CPU   Iterations
--------------------------------------------------------------------------
CPU-Naive/8/manual_time              0.409 us        0.435 us      1709947
CPU-Naive/16/manual_time              2.97 us         3.00 us       235301
CPU-Naive/32/manual_time              28.5 us         28.5 us        24584
CPU-Naive/64/manual_time               230 us          230 us         3045
CPU-Naive/128/manual_time             2109 us         2109 us          331
CPU-Naive/256/manual_time            18478 us        18475 us           38
CPU-Naive/512/manual_time           179765 us       179738 us            4
CPU-Naive/1024/manual_time         3094298 us      3093892 us            1
CPU-Naive/2048/manual_time        40399168 us     40393110 us            1
CPU-Naive/manual_time_BigO      15378613.70 N    15376318.01 N    
CPU-Naive/manual_time_RMS              123 %           123 %    
CPU-NaiveTrans/8/manual_time         0.441 us        0.468 us      1589424
CPU-NaiveTrans/16/manual_time         2.98 us         3.01 us       234761
CPU-NaiveTrans/32/manual_time         26.2 us         26.2 us        27156
CPU-NaiveTrans/64/manual_time          231 us          231 us         2979
CPU-NaiveTrans/128/manual_time        2112 us         2111 us          332
CPU-NaiveTrans/256/manual_time       18071 us        18070 us           39
CPU-NaiveTrans/512/manual_time      150437 us       150427 us            5
CPU-NaiveTrans/1024/manual_time    1217076 us      1216933 us            1
CPU-NaiveTrans/2048/manual_time    9908246 us      9906791 us            1
CPU-NaiveTrans/manual_time_BigO 3866027.38 N    3865467.44 N    
CPU-NaiveTrans/manual_time_RMS         107 %           107 %    
CPU-OpenMP/8/manual_time              12.2 us         12.3 us        57873
CPU-OpenMP/16/manual_time             20.3 us         20.3 us        38720
CPU-OpenMP/32/manual_time             58.7 us         58.9 us        10650
CPU-OpenMP/64/manual_time             34.4 us         34.5 us        19373
CPU-OpenMP/128/manual_time             113 us          113 us         5906
CPU-OpenMP/256/manual_time             705 us          705 us         1034
CPU-OpenMP/512/manual_time            6724 us         6724 us          103
CPU-OpenMP/1024/manual_time         214191 us       209030 us            3
CPU-OpenMP/2048/manual_time        2162594 us      2152563 us            1
CPU-OpenMP/4096/manual_time       35601416 us     33953207 us            1
CPU-OpenMP/manual_time_BigO     6726779.47 N    6423828.56 N    
CPU-OpenMP/manual_time_RMS             134 %           133 %    
CPU-OMP-SSE/8/manual_time             23.8 us         23.9 us        29143
CPU-OMP-SSE/16/manual_time            29.2 us         29.3 us        23323
CPU-OMP-SSE/32/manual_time            47.0 us         47.2 us        14987
CPU-OMP-SSE/64/manual_time             107 us          107 us         6459
CPU-OMP-SSE/128/manual_time            380 us          380 us         1779
CPU-OMP-SSE/256/manual_time            300 us          300 us         2330
CPU-OMP-SSE/512/manual_time           1369 us         1369 us          515
CPU-OMP-SSE/1024/manual_time          8813 us         8813 us           75
CPU-OMP-SSE/2048/manual_time         83183 us        83160 us            8
CPU-OMP-SSE/4096/manual_time        852229 us       852078 us            1
CPU-OMP-SSE/8192/manual_time       7523373 us      7518819 us            1
CPU-OMP-SSE/manual_time_BigO     729811.55 N     729387.19 N    
CPU-OMP-SSE/manual_time_RMS            122 %           122 %    
CPU-OMP-AVX/8/manual_time             23.7 us         23.8 us        29644
CPU-OMP-AVX/16/manual_time            28.7 us         28.8 us        24670
CPU-OMP-AVX/32/manual_time            47.8 us         48.0 us        14901
CPU-OMP-AVX/64/manual_time             130 us          131 us         5260
CPU-OMP-AVX/128/manual_time            433 us          433 us         1573
CPU-OMP-AVX/256/manual_time            243 us          243 us         2884
CPU-OMP-AVX/512/manual_time           1021 us         1021 us          690
CPU-OMP-AVX/1024/manual_time          6402 us         6402 us          110
CPU-OMP-AVX/2048/manual_time         66029 us        66029 us           10
CPU-OMP-AVX/4096/manual_time        860487 us       763444 us            1
CPU-OMP-AVX/8192/manual_time       6978794 us      6922671 us            1
CPU-OMP-AVX/manual_time_BigO     679909.58 N     670329.11 N    
CPU-OMP-AVX/manual_time_RMS            119 %           123 %    
GPU-Naive/8/manual_time               6.82 us         6.84 us       102368
GPU-Naive/16/manual_time              7.14 us         7.17 us        97921
GPU-Naive/32/manual_time              7.74 us         7.77 us        89738
GPU-Naive/64/manual_time              7.94 us         7.97 us        79574
GPU-Naive/128/manual_time             10.1 us         10.2 us        69024
GPU-Naive/256/manual_time             33.9 us         33.9 us        20685
GPU-Naive/512/manual_time              219 us          219 us         3191
GPU-Naive/1024/manual_time            1652 us         1651 us          424
GPU-Naive/2048/manual_time           14217 us        14215 us           49
GPU-Naive/4096/manual_time          114310 us       114281 us            6
GPU-Naive/8192/manual_time          973652 us       973499 us            1
GPU-Naive/manual_time_BigO        94718.86 N      94703.48 N    
GPU-Naive/manual_time_RMS              120 %           120 %    
GPU-Shared/8/manual_time              6.19 us         6.22 us       112940
GPU-Shared/16/manual_time             6.61 us         6.63 us       105819
GPU-Shared/32/manual_time             7.63 us         7.65 us        91814
GPU-Shared/64/manual_time             9.35 us         9.38 us        74934
GPU-Shared/128/manual_time            12.8 us         12.8 us        54720
GPU-Shared/256/manual_time            22.4 us         22.4 us        31290
GPU-Shared/512/manual_time             131 us          131 us         5362
GPU-Shared/1024/manual_time            996 us          996 us          703
GPU-Shared/2048/manual_time           7971 us         7971 us           88
GPU-Shared/4096/manual_time          65896 us        65886 us           11
GPU-Shared/8192/manual_time         569202 us       569092 us            1
GPU-Shared/manual_time_BigO       55323.22 N      55312.67 N    
GPU-Shared/manual_time_RMS             120 %           120 % 
```

![matmul_benchmark](figures/matmul_benchmark.png)

