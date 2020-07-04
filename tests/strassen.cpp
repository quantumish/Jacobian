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
Eigen::MatrixXd strassen_mul(Eigen::MatrixXd x, Eigen::MatrixXd y)
{
  int power = 0;
  int largest;
  if (x.rows() >= x.cols() || x.rows() >= y.cols() || x.rows() >= y.rows()) largest = x.rows();
  else if (x.cols() >= x.rows() || x.cols() >= y.cols() || x.cols() >= y.rows()) largest = x.cols();
  else if (y.cols() >= x.rows() || y.cols() >= x.cols() || y.cols() >= y.rows()) largest = y.cols();
  else if (y.rows() >= x.rows() || y.rows() >= x.cols() || y.rows() >= y.cols()) largest = y.rows();
  for (; pow(2, power) < largest; power++) {
    std::cout << pow(2, power) << " " << power <<"\n";
  }
  std::cout << "POWER: " << power << " LARGEST: " << largest << "\n";
  Eigen::MatrixXd a ((int)pow(2,power), (int)pow(2,power));
  Eigen::MatrixXd b ((int)pow(2,power), (int)pow(2,power));
  a.block(0,0,x.rows(), x.cols()) = x;
  b.block(0,0,y.rows(), y.cols()) = y;
  //Eigen::MatrixXd a ()
  int block_len = largest/2;
  Eigen::MatrixXd result (a.rows(), a.cols());

  Eigen::MatrixXd m1 = ((a.block(0,0, block_len, block_len)) + a.block(a.rows()-block_len,a.cols()-block_len, block_len, block_len)) * (b.block(0,0, block_len, block_len) + b.block(b.rows()-block_len,b.cols()-block_len, block_len, block_len));
  Eigen::MatrixXd m2 = (a.block(a.rows()-block_len, 0, block_len, block_len) + a.block(a.rows()-block_len,a.cols()-block_len, block_len, block_len)) * (b.block(0,0, block_len, block_len));
  Eigen::MatrixXd m3 = a.block(0,0, block_len, block_len) * (b.block(0,b.cols()-block_len, block_len, block_len) - b.block(b.rows()-block_len,b.cols()-block_len, block_len, block_len));
  Eigen::MatrixXd m4 = a.block(a.rows()-block_len,a.cols()-block_len, block_len, block_len) * (b.block(b.rows()-block_len,0, block_len, block_len) - b.block(0,0, block_len, block_len));
  Eigen::MatrixXd m5 = (a.block(0, 0, block_len, block_len) + a.block(0,a.cols()-block_len, block_len, block_len)) * (b.block(b.rows()-block_len,b.cols()-block_len, block_len, block_len));
  Eigen::MatrixXd m6 = (a.block(a.rows()-block_len,0, block_len, block_len) - a.block(0,0, block_len, block_len)) * (b.block(0,0, block_len, block_len) + b.block(0,b.cols()-block_len, block_len, block_len));
  Eigen::MatrixXd m7 = (a.block(0,a.cols()-block_len, block_len, block_len) - a.block(a.rows()-block_len,a.cols()-block_len, block_len, block_len)) * (b.block(a.rows()-block_len,0, block_len, block_len) + b.block(b.rows()-block_len,b.cols()-block_len, block_len, block_len));

  
  result.block(0,0, block_len, block_len) =  m1 + m4 - m5 + m7;
  result.block(0,result.cols()-block_len, block_len, block_len) =  m3 + m5;
  result.block(result.rows()-block_len,0, block_len, block_len) =  m2 + m4;
  result.block(result.rows()-block_len,result.cols()-block_len, block_len, block_len) =  m1 -m2 + m3 + m6;

  return result;
}

int main()
{
  srand((unsigned int) time(0));
  int sz;
  std::cin >> sz;
  Eigen::MatrixXd a = Eigen::MatrixXd::Random(sz, 3);
  Eigen::MatrixXd b = Eigen::MatrixXd::Random(3, sz);
  
  auto eigen_begin = std::chrono::high_resolution_clock::now();
  Eigen::MatrixXd product = a * b;
  auto eigen_end = std::chrono::high_resolution_clock::now();
  
  auto strassen_begin = std::chrono::high_resolution_clock::now();
  Eigen::MatrixXd sproduct = strassen_mul(a, b);
  auto strassen_end = std::chrono::high_resolution_clock::now();
  
  std::cout << "EIGEN: " << std::chrono::duration_cast<std::chrono::nanoseconds>(eigen_end - eigen_begin).count() / pow(10,9) << " STRASSEN: " << std::chrono::duration_cast<std::chrono::nanoseconds>(strassen_end - strassen_begin).count() / pow(10,9) << "\n";
  std::cout << "A:\n" << a << "\nB:\n" << b << "\nEigen:\n" << product << "\nStrassen:\n" << sproduct << "\n";
}
