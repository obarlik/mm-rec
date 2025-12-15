// Eigen Matmul Demo (same benchmark)
#include <Eigen/Dense>
#include <iostream>
#include <chrono>

void benchmark_eigen(int N) {
    Eigen::MatrixXf A = Eigen::MatrixXf::Ones(N, N);
    Eigen::MatrixXf B = Eigen::MatrixXf::Ones(N, N);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Eigen matmul (automatically optimized)
    Eigen::MatrixXf C = A * B;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Eigen: " << N << "x" << N << " matmul: " 
              << duration.count() / 1000.0 << " ms" << std::endl;
}

int main() {
    std::cout << "=== Eigen Performance Test ===" << std::endl;
    benchmark_eigen(100);
    benchmark_eigen(500);
    benchmark_eigen(1000);
    return 0;
}
