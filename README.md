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

- Strassen : 


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
Load Average: 0.25, 4.03, 5.77
--------------------------------------------------------------------------
Benchmark                                Time             CPU   Iterations
--------------------------------------------------------------------------
CPU-Naive/8/manual_time              0.368 us        0.395 us      1897394
CPU-Naive/16/manual_time              2.78 us         2.81 us       250596
CPU-Naive/32/manual_time              24.3 us         24.3 us        28841
CPU-Naive/64/manual_time               227 us          227 us         3088
CPU-Naive/128/manual_time             2095 us         2095 us          334
CPU-Naive/256/manual_time            18374 us        18374 us           38
CPU-Naive/512/manual_time           196500 us       196493 us            3
CPU-Naive/1024/manual_time         3123693 us      3123556 us            1
CPU-Naive/2048/manual_time        41510380 us     41507821 us            1
CPU-Naive/manual_time_BigO      15792462.70 N    15791499.86 N    
CPU-Naive/manual_time_RMS              123 %           123 %    
CPU-NaiveTrans/8/manual_time         0.432 us        0.459 us      1620679
CPU-NaiveTrans/16/manual_time         2.91 us         2.93 us       240960
CPU-NaiveTrans/32/manual_time         25.0 us         25.1 us        27869
CPU-NaiveTrans/64/manual_time          225 us          225 us         3113
CPU-NaiveTrans/128/manual_time        2085 us         2085 us          336
CPU-NaiveTrans/256/manual_time       18001 us        18000 us           39
CPU-NaiveTrans/512/manual_time      149415 us       149410 us            5
CPU-NaiveTrans/1024/manual_time    1218460 us      1218368 us            1
CPU-NaiveTrans/2048/manual_time    9895883 us      9894650 us            1
CPU-NaiveTrans/manual_time_BigO 3861655.82 N    3861186.72 N    
CPU-NaiveTrans/manual_time_RMS         106 %           106 %    
CPU-OpenMP/8/manual_time              9.05 us         9.19 us        77347
CPU-OpenMP/16/manual_time             10.4 us         10.6 us        71144
CPU-OpenMP/32/manual_time             11.9 us         12.1 us        57979
CPU-OpenMP/64/manual_time             21.5 us         21.7 us        32430
CPU-OpenMP/128/manual_time            83.5 us         83.6 us         8291
CPU-OpenMP/256/manual_time             640 us          640 us         1088
CPU-OpenMP/512/manual_time            6572 us         6572 us           98
CPU-OpenMP/1024/manual_time         214711 us       214707 us            3
CPU-OpenMP/2048/manual_time        2178299 us      2110478 us            1
CPU-OpenMP/4096/manual_time       53092101 us     46705942 us            1
CPU-OpenMP/manual_time_BigO     9930879.70 N    8755328.66 N    
CPU-OpenMP/manual_time_RMS             142 %           140 %    
CPU-OMP-SSE/8/manual_time             17.3 us         17.4 us        40274
CPU-OMP-SSE/16/manual_time            21.3 us         21.4 us        32249
CPU-OMP-SSE/32/manual_time            27.0 us         27.1 us        26222
CPU-OMP-SSE/64/manual_time            44.3 us         44.4 us        17181
CPU-OMP-SSE/128/manual_time           79.2 us         79.3 us         8801
CPU-OMP-SSE/256/manual_time            135 us          135 us         4586
CPU-OMP-SSE/512/manual_time            471 us          471 us         1500
CPU-OMP-SSE/1024/manual_time          2553 us         2553 us          274
CPU-OMP-SSE/2048/manual_time         24025 us        24016 us           27
CPU-OMP-SSE/4096/manual_time        333261 us       328200 us            2
CPU-OMP-SSE/8192/manual_time       1964406 us      1921983 us            1
CPU-OMP-SSE/manual_time_BigO     195684.60 N     191568.78 N    
CPU-OMP-SSE/manual_time_RMS            105 %           105 %    
CPU-OMP-AVX/8/manual_time             20.1 us         20.2 us        35094
CPU-OMP-AVX/16/manual_time            22.0 us         22.2 us        31590
CPU-OMP-AVX/32/manual_time            27.1 us         27.2 us        25605
CPU-OMP-AVX/64/manual_time            42.0 us         42.1 us        16754
CPU-OMP-AVX/128/manual_time           83.7 us         83.8 us         8515
CPU-OMP-AVX/256/manual_time            119 us          119 us         5813
CPU-OMP-AVX/512/manual_time            301 us          301 us         2244
CPU-OMP-AVX/1024/manual_time          1257 us         1257 us          554
CPU-OMP-AVX/2048/manual_time         13128 us        13127 us           51
CPU-OMP-AVX/4096/manual_time        194800 us       191627 us            3
CPU-OMP-AVX/8192/manual_time       1179536 us      1178624 us            1
CPU-OMP-AVX/manual_time_BigO     117224.11 N     116995.34 N    
CPU-OMP-AVX/manual_time_RMS            107 %           107 %    
GPU-Naive/8/manual_time               6.49 us         6.51 us       108212
GPU-Naive/16/manual_time              6.75 us         6.78 us       103692
GPU-Naive/32/manual_time              7.20 us         7.23 us        93594
GPU-Naive/64/manual_time              7.37 us         7.39 us        87746
GPU-Naive/128/manual_time             9.60 us         9.63 us        72716
GPU-Naive/256/manual_time             33.6 us         33.6 us        20979
GPU-Naive/512/manual_time              219 us          219 us         3198
GPU-Naive/1024/manual_time            1652 us         1652 us          424
GPU-Naive/2048/manual_time           14210 us        14209 us           50
GPU-Naive/4096/manual_time          114234 us       114228 us            6
GPU-Naive/8192/manual_time          950395 us       950293 us            1
GPU-Naive/manual_time_BigO        92585.98 N      92576.37 N    
GPU-Naive/manual_time_RMS              119 %           119 %    
GPU-Shared/8/manual_time              5.79 us         5.82 us       120341
GPU-Shared/16/manual_time             6.23 us         6.26 us       112511
GPU-Shared/32/manual_time             7.26 us         7.28 us        94649
GPU-Shared/64/manual_time             8.98 us         9.00 us        77702
GPU-Shared/128/manual_time            12.4 us         12.5 us        56268
GPU-Shared/256/manual_time            22.1 us         22.1 us        31783
GPU-Shared/512/manual_time             130 us          130 us         5374
GPU-Shared/1024/manual_time            992 us          992 us          709
GPU-Shared/2048/manual_time           7970 us         7969 us           88
GPU-Shared/4096/manual_time          68307 us        68306 us           11
GPU-Shared/8192/manual_time         560628 us       560590 us            1
GPU-Shared/manual_time_BigO       54648.48 N      54644.96 N    
GPU-Shared/manual_time_RMS             118 %           118 %   
```

![matmul_benchmark](figures/matmul_benchmark.png)

