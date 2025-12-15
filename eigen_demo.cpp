// Eigen Demo - ZERO Runtime Dependency Proof
#include <Eigen/Dense>
#include <iostream>

int main() {
    // Matrix multiplication with Eigen (header-only!)
    Eigen::MatrixXf A = Eigen::MatrixXf::Random(100, 100);
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(100, 100);
    
    // Automatically optimized with SIMD
    Eigen::MatrixXf C = A * B;
    
    std::cout << "✅ Eigen matmul works!" << std::endl;
    std::cout << "   Matrix size: 100x100" << std::endl;
    std::cout << "   Result[0,0]: " << C(0,0) << std::endl;
    std::cout << "\n🎉 NO RUNTIME DEPENDENCIES (header-only!)" << std::endl;
    
    return 0;
}
