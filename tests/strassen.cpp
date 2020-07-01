//
// strassen.cpp
// Experimental benchmarking of Strassen's Algorithm vs plain Eigen matrix multipy
//
// While naive multiplication is O(n^3), Strassen's Algorithm for matrix multiply is O(n^2.8074)
// which means significant differences will begin to manifest for large matrices. All faster algorithms are galactic.
//

#include <Eigen/Dense>

// Not extensible, just initial test.
Eigen::MatrixXd strassen_mul(Eigen::MatrixXd a, Eigen::MatrixXd b)
{
  block_len = a.rows()/4;
}

Eigen::MatrixXd a = Eigen::MatrixXd::Random(5000, 5000);
Eigen::MatrixXd b = Eigen::MatrixXd::Random(5000, 5000);

auto eigen_begin = std::chrono::high_resolution_clock::now();
Eigen::MatrixXd product = a * b;
auto eigen_end = std::chrono::high_resolution_clock::now();

auto strassen_begin = std::chrono::high_resolution_clock::now();
auto strassen_end = std::chrono::high_resolution_clock::now();
