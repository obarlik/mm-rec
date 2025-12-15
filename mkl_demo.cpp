// MKL Matmul Demo
#include <mkl.h>
#include <iostream>
#include <chrono>
#include <vector>

void benchmark_mkl(int N) {
    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 1.0f);
    std::vector<float> C(N * N, 0.0f);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // MKL matmul: C = A * B
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        N, N, N,
        1.0f, A.data(), N,
        B.data(), N,
        0.0f, C.data(), N
    );
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "MKL: " << N << "x" << N << " matmul: " 
              << duration.count() / 1000.0 << " ms" << std::endl;
}

int main() {
    std::cout << "=== MKL Performance Test ===" << std::endl;
    benchmark_mkl(100);
    benchmark_mkl(500);
    benchmark_mkl(1000);
    return 0;
}
