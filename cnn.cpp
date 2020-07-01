#include "bpnn.hpp"
#include "utils.hpp"

#define LARGE_NUM = 1000000 // Remove me.

class ConvLayer
{
  Eigen::MatrixXd* input;
  Eigen::MatrixXd* kernel;
  Eigen::MatrixXd* output;

public:
  ConvLayer(int stride) 
  void convolute();
};

void ConvLayer::convolute()
{
  for (int i = 0; i < input->cols() - kernel->cols() + 1; i+=stride_len) {
    for (int j = 0; j < input->rows() - kernel->rows() + 1; j+=stride_len) {
      output(j, i) = (*kernel * input->block<kernel->rows(), kernel->cols()(j, i)).sum()
    }
  }
}

class PoolingLayer
{
  Eigen::MatrixXd* input;
  Eigen::MatrixXd* kernel;
  Eigen::MatrixXd* output;

public:
  void pool();
};

void PoolingLayer::pool()
{
  // It doesn't look like anything better than O(n^4) is doable for this as kernel needs to go through matrix and you need to index kernel. LOOK INTO ME!! 
  float maxnum = -LARGE_NUM;
  for (int i = 0; i < input->cols() - kernel->cols() + 1; i+=stride_len) {
    for (int j = 0; j < input->rows() - kernel->rows() + 1; j+=stride_len) {
      for (int k = 0; k < kernel->cols(); k++) {
        for (int l = 0; l < kernel->rows(); l++) {
          if ((input->block<kernel->rows(), kernel->cols()(j, i))(l, k) > maxnum) {
            maxnum = (input->block<kernel->rows(), kernel->cols()(j, i))(l, k);
          }
        }
      }
    }
  }
}

class ConvNet : Network
{
  int stride_len;
  int preprocess_length;
                
  std::vector<ConvLayer> conv_layers;
  std::vector<PoolingLayer> pool_layers;
public:
  ConvNet(char* path, int batch_sz, float learn_rate, float bias_rate);
  void process(); // Runs the convolutional and pooling layers.
  void next_batch();
  void add_conv_layer();
};

ConvNet::ConvNet(char* path, int batch_sz, float learn_rate, float bias_rate)
{
  learning_rate = learn_rate;
  bias_lr = bias_rate;
  instances = prep_file(path, "./shuffled.txt");
  length = 0;
  t = 0;
  batch_size = batch_sz;
  data = fopen("./shuffled.txt", "r");
  batches = 0;
}
