

nvcc kernel.cu -c -std=c++11 -O3 -D_FORCE_INLINES &&\
g++ kernel.o matmul.cpp benchmark.cpp -o bench -std=c++11 -lpthread -fopenmp -lbenchmark -lcudart -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -mavx -O3 &&\
rm kernel.o && \
echo "compile finished!"
