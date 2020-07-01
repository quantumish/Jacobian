//
// strassen.cpp
// Experimental benchmarking of Strassen's Algorithm vs plain Eigen matrix multipy
//
// While naive multiplication is O(n^3), Strassen's Algorithm for matrix multiply is O(n^2.8074)
// which means significant differences will begin to manifest for large matrices. All faster algorithms are galactic.
//

#include <Eigen/Dense>
#include <ctime>
#include <iostream>

// Not remotely extensible, just initial test.
Eigen::MatrixXd strassen_mul(Eigen::MatrixXd a, Eigen::MatrixXd b)
{
  int block_len = a.rows()/2;
  Eigen::MatrixXd result (a.rows(), a.cols());

  // (A+D)(E+H)
  Eigen::MatrixXd m1 = (a.block(0,0, block_len, block_len)) + a.block(a.rows()-block_len,a.cols()-block_len, block_len, block_len) * (b.block(0,0, block_len, block_len) + b.block(b.rows()-block_len,b.cols()-block_len, block_len, block_len));
  // (C+D)E
  Eigen::MatrixXd m2 = (a.block(a.rows()-block_len, 0, block_len, block_len) + a.block(a.rows()-block_len,a.cols()-block_len, block_len, block_len)) * (b.block(0,0, block_len, block_len));
  // A(F-H)
  Eigen::MatrixXd m3 = a.block(0,0, block_len, block_len) * (b.block(0,b.cols()-block_len, block_len, block_len) - b.block(b.rows()-block_len,b.cols()-block_len, block_len, block_len));
  // D(G-E)
  Eigen::MatrixXd m4 = a.block(a.rows()-block_len,a.cols()-block_len, block_len, block_len) * (b.block(b.rows()-block_len,0, block_len, block_len) - b.block(0,0, block_len, block_len));
  // (A+B)H
  Eigen::MatrixXd m5 = (a.block(0, 0, block_len, block_len) + a.block(0,a.cols()-block_len, block_len, block_len)) * (b.block(b.rows()-block_len,b.cols()-block_len, block_len, block_len));
  // (A-C)(E+F)
  Eigen::MatrixXd m6 = (a.block(0,0, block_len, block_len) - a.block(a.rows()-block_len,0, block_len, block_len)) * (b.block(0,0, block_len, block_len) + b.block(0,b.cols()-block_len, block_len, block_len));
  // (B-D)(G+H)
  Eigen::MatrixXd m7 = (a.block(0,a.cols()-block_len, block_len, block_len) - a.block(a.rows()-block_len,a.cols()-block_len, block_len, block_len)) * (b.block(a.rows()-block_len,0, block_len, block_len) + b.block(b.rows()-block_len,b.cols()-block_len, block_len, block_len));

  result.block(0,0, block_len, block_len) =  m1 + m4 - m5 + m7;
  result.block(0,result.cols()-block_len, block_len, block_len) =  m3 + m5;
  result.block(result.rows()-block_len,0, block_len, block_len) =  m2 + m4;
  result.block(result.rows()-block_len,result.cols()-block_len, block_len, block_len) =  m1 - m6 + m3 - m2;

  return result;
}

int main()
{
  Eigen::MatrixXd a = Eigen::MatrixXd::Random(16, 16);
  Eigen::MatrixXd b = Eigen::MatrixXd::Random(16, 16);
  
  auto eigen_begin = std::chrono::high_resolution_clock::now();
  Eigen::MatrixXd product = a * b;
  auto eigen_end = std::chrono::high_resolution_clock::now();
  
  auto strassen_begin = std::chrono::high_resolution_clock::now();
  Eigen::MatrixXd sproduct = strassen_mul(a, b);
  auto strassen_end = std::chrono::high_resolution_clock::now();
  
  std::cout << "EIGEN: " << std::chrono::duration_cast<std::chrono::nanoseconds>(eigen_end - eigen_begin).count() / pow(10,9) << " STRASSEN: " << std::chrono::duration_cast<std::chrono::nanoseconds>(strassen_end - strassen_begin).count() / pow(10,9) << "\n";
  std::cout << "A:\n" << a << "\nB:\n" << b << "\nEigen:\n" << product << "\nStrassen:\n" << sproduct << "\n";
}
