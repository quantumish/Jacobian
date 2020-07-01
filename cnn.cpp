#include "bpnn.hpp"
#include "utils.hpp"

class ConvLayer
{
  Eigen::MatrixXd* input;
  Eigen::MatrixXd* kernel;
  Eigen::MatrixXd* output;
};
class PoolingLayer
{
};

class ConvNet : Network
{
public:
  ConvNet(char* path, int batch_sz, float learn_rate, float bias_rate);
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
