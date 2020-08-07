#include <Eigen/Dense>
#include <Eigen/MatrixFunctions>

#include <iostream>

std::complex<float> lecun_tanh(std::complex<float> x, int) {return (float)1.7159 * tanh(((float)2.0/3) * x);}
//std::complex<float> lecun_tanh_deriv(std::complex<float> x, int) {return 1.14393 * pow(1.0/cosh(2.0/3 * x),2);}

std::complex<float> mat_sigmoid(std::complex<float> x, int) {return (float)1.0/((float)1+exp(-x));}
float sigmoid(float x) {return 1.0/(1+exp(-x));}

int main()
{
  int size;
  std::cin >> size;
  Eigen::MatrixXf m = Eigen::MatrixXf::Random(size,size);
  auto mat_start = std::chrono::high_resolution_clock::now();
  m = m.matrixFunction(mat_sigmoid);
  auto mat_end = std::chrono::high_resolution_clock::now();
  Eigen::MatrixXf m2 = Eigen::MatrixXf::Random(size,size);
  auto start = std::chrono::high_resolution_clock::now();
  for (int j = 0; j < m2.rows(); j++) {
    for (int k = 0; k < m2.cols(); k++) {
      m2(j,k) = sigmoid(m2(j,k));
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "MATRIX: " << std::chrono::duration_cast<std::chrono::nanoseconds>(mat_end - mat_start).count() << " NORMAL: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << "\n";

}


