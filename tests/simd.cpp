#include <Eigen/Dense>
#include <ctime>
#include <iostream>
#include <cmath>

int main()
{
  Eigen::MatrixXf a = Eigen::MatrixXf::Random(1000, 1000);

  auto simd_start = std::chrono::high_resolution_clock::now();
  a.array().exp().matrix();
  auto simd_end = std::chrono::high_resolution_clock::now();
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < a.rows(); i++) {
    for (int j = 0; j < a.cols(); j++) {
      a(i,j) = exp(a(i,j));
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "SIMD: " << std::chrono::duration_cast<std::chrono::nanoseconds>(simd_end - simd_start).count() << " NORMAL: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << "\n";
}
