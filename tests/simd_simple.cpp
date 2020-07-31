#include <Eigen/Dense>
#include <ctime>
#include <iostream>
#include <cmath>

float sigmoid(float x) {return 1.0/(1+exp(-x));}
float sigmoid_deriv(float x) {return 1.0/(1+exp(-x)) * (1 - 1.0/(1+exp(-x)));}

int main()
{
  int sz;
  std::cin >> sz;
  Eigen::MatrixXf a = Eigen::MatrixXf::Random(1, sz);
  Eigen::MatrixXf::Index maxRow, maxCol;
  float max = a.maxCoeff(&maxRow, &maxCol);
  Eigen::MatrixXf m1 = ((a.array() - max).exp() / ((a.array() - max).exp().sum())).matrix();
  auto softmax_simd_start = std::chrono::high_resolution_clock::now();
  Eigen::MatrixXf a1 = (1 + (-1 * a.array()).exp()).pow(-1).matrix();
  auto softmax_simd_end = std::chrono::high_resolution_clock::now();
  auto softmax_start = std::chrono::high_resolution_clock::now();
  // float sum = 0;
  Eigen::MatrixXf::Index maxRow2, maxCol2;
  float max2 = a.maxCoeff(&maxRow2, &maxCol2);
  Eigen::MatrixXf m2 = (a.array() - max2).matrix();
  // for (int j = 0; j < m2.cols(); j++) {
  //   sum += exp(m2(0,j));
  // }
  float sum = ((a.array() - max2).exp().sum());
  for (int j = 0; j < m2.cols(); j++) {
    m2(0,j) = exp(m2(0,j))/sum;
  }
  auto softmax_end = std::chrono::high_resolution_clock::now();
  if (m1.isApprox(m2)) std::cout << "Softmax matrices are equal!\n\n";

  double eigen_time = std::chrono::duration_cast<std::chrono::nanoseconds>(softmax_simd_end - softmax_simd_start).count();
  double normal_time = std::chrono::duration_cast<std::chrono::nanoseconds>(softmax_end - softmax_start).count();
  std::cout << "SOFTMAX:\n\tEIGEN FUNCTIONS (SOME SIMD): " << eigen_time << "\n\tNORMAL ITERATIVE METHOD: " << normal_time << "\n\tSPEEDUP FACTOR: " << normal_time/eigen_time << "\n";
}
