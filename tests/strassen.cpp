//
// strassen.cpp
// Experimental benchmarking of Strassen's Algorithm vs plain Eigen matrix multipy
//
// While naive multiplication is O(n^3), Strassen's Algorithm for matrix multiply is O(n^2.8074)
// which means significant differences will begin to manifest for large matrices. All faster algorithms are galactic.
//

#include <Eigen/Dense>

// Not remotely extensible, just initial test.
Eigen::MatrixXd strassen_mul(Eigen::MatrixXd a, Eigen::MatrixXd b)
{
  int block_len = a.rows()/4;
  Eigen::MatrixXd result (a.rows(), a.cols());
  // Nigh unreadable code follows.
  Eigen::MatrixXd m1 = (a.block<block_len, block_len>(0,0) + a.block<block_len, block_len>(a.rows()-block_len,a.cols()-block_len)) * (b.block<block_len, block_len>(0,0) + b.block<block_len, block_len>(b.rows()-block_len,b.cols()-block_len));
  Eigen::MatrixXd m2 = (a.block<block_len, block_len>(a.rows()-block_len, 0) + a.block<block_len, block_len>(a.rows()-block_len,a.cols()-block_len)) * (b.block<block_len, block_len>(0,0));
  Eigen::MatrixXd m3 = a.block<block_len, block_len>(0,0) * (b.block<block_len, block_len>(0,b.cols()-block_len) - b.block<block_len, block_len>(b.rows()-block_len,b.cols()-block_len));
  Eigen::MatrixXd m4 = a.block<block_len, block_len>(a.rows()-block_len,a.cols()-block_len) * (b.block<block_len, block_len>(b.rows()-block_len,0) - b.block<block_len, block_len>(0,0));
  Eigen::MatrixXd m5 = (a.block<block_len, block_len>(0, 0) + a.block<block_len, block_len>(0,a.cols()-block_len)) * (b.block<block_len, block_len>(b.rows()-block_len,b.cols()-block_len));
  Eigen::MatrixXd m6 = (a.block<block_len, block_len>(a.rows()-block_len,0) - a.block<block_len, block_len>(0,0)) * (b.block<block_len, block_len>(0,0) + b.block<block_len, block_len>(0,b.cols()-block_len));
  Eigen::MatrixXd m7 = (a.block<block_len, block_len>(0,a.cols()-block_len) - a.block<block_len, block_len>(a.rows()-block_len,a.cols()-block_len)) * (b.block<block_len, block_len>(a.rows()-block_len,0) + b.block<block_len, block_len>(b.rows()-block_len,b.cols()-block_len));

  result.block<block_len, block_len>(0,0) =  m1 + m4 - m5 + m7;
  result.block<block_len, block_len>(0,results.cols()-block_len) =  m3 + m5;
  result.block<block_len, block_len>(results.rows()-block_len,0) =  m2 + m4;
  result.block<block_len, block_len>(results.rows()-block_len,result.cols()-block_len) =  m1 - m2 + m3 + m6;
  return result
}

Eigen::MatrixXd a = Eigen::MatrixXd::Random(5000, 5000);
Eigen::MatrixXd b = Eigen::MatrixXd::Random(5000, 5000);

auto eigen_begin = std::chrono::high_resolution_clock::now();
Eigen::MatrixXd product = a * b;
auto eigen_end = std::chrono::high_resolution_clock::now();

auto strassen_begin = std::chrono::high_resolution_clock::now();
Eigen::MatrixXd sproduct = strassen_mul(a, b);
auto strassen_end = std::chrono::high_resolution_clock::now();

std::cout << "EIGEN: " << std::chrono::duration_cast<std::chrono::nanoseconds>(eigen_end - eigen_begin) << " STRASSEN: " << std::chrono::duration_cast<std::chrono::nanoseconds>(strassen_end - strassen_begin);
