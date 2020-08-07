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
  Eigen::MatrixXf a = Eigen::MatrixXf::Random(sz, sz);

  auto sigmoid_simd_start = std::chrono::high_resolution_clock::now();
  Eigen::MatrixXf a1 = (1 + (-1 * a.array()).exp()).pow(-1).matrix();
  auto sigmoid_simd_end = std::chrono::high_resolution_clock::now();
  auto sigmoid_start = std::chrono::high_resolution_clock::now();
  Eigen::MatrixXf a2 (a.rows(), a.cols());
  for (int i = 0; i < a.rows(); i++) {
    for (int j = 0; j < a.cols(); j++) {
      a2(i,j) = sigmoid(a(i,j));
    }
  }
  auto sigmoid_end = std::chrono::high_resolution_clock::now();
  if (a1.isApprox(a2)) std::cout << "Sigmoid matrices are equal!\n\n";

  auto sigmoid_deriv_simd_start = std::chrono::high_resolution_clock::now();
  Eigen::MatrixXf ones = Eigen::MatrixXf::Ones(a.rows(), a.cols());
  Eigen::MatrixXf deriv_a1 = (1 + (-1 * a.array()).exp()).pow(-1).matrix().cwiseProduct(ones-(1 + (-1 * a.array()).exp()).pow(-1).matrix());
  auto sigmoid_deriv_simd_end = std::chrono::high_resolution_clock::now();
  auto sigmoid_deriv_start = std::chrono::high_resolution_clock::now();
  Eigen::MatrixXf deriv_a2 (a.rows(), a.cols());
  for (int i = 0; i < a.rows(); i++) {
    for (int j = 0; j < a.cols(); j++) {
      deriv_a2(i,j) = sigmoid_deriv(a(i,j));
    }
  }
  auto sigmoid_deriv_end = std::chrono::high_resolution_clock::now();
  if (deriv_a1.isApprox(deriv_a2)) std::cout << "Sigmoid derivative matrices are equal!\n\n";

  double eigen_time = std::chrono::duration_cast<std::chrono::nanoseconds>(sigmoid_simd_end - sigmoid_simd_start).count();
  double normal_time = std::chrono::duration_cast<std::chrono::nanoseconds>(sigmoid_end - sigmoid_start).count();
  double eigen_deriv_time = std::chrono::duration_cast<std::chrono::nanoseconds>(sigmoid_deriv_simd_end - sigmoid_deriv_simd_start).count();
  double normal_deriv_time = std::chrono::duration_cast<std::chrono::nanoseconds>(sigmoid_deriv_end - sigmoid_deriv_start).count();
  std::cout << "SIGMOID:\n\tEIGEN FUNCTIONS (SOME SIMD): " << eigen_time << "\n\tNORMAL ITERATIVE METHOD: " << normal_time << "\n\tSPEEDUP FACTOR: " << normal_time/eigen_time << "\n";
  std::cout << "SIGMOID DERIVATIVE:\n\tEIGEN FUNCTIONS (SOME SIMD): " << eigen_deriv_time << "\n\tNORMAL ITERATIVE METHOD: " << normal_deriv_time << "\n\tSPEEDUP FACTOR: " << normal_deriv_time/eigen_deriv_time << "\n";
}
