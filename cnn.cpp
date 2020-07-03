#include "bpnn.hpp"
#include "utils.hpp"

#define LARGE_NUM 1000000 // Remove me.

class ConvLayer
{
public:
  int stride_len;
  Eigen::MatrixXd* input;
  Eigen::MatrixXd* kernel;
  Eigen::MatrixXd* output;
  
  ConvLayer(int x, int y, int stride, int kernel_size);
  void convolute();
};

ConvLayer::ConvLayer(int x, int y, int stride, int kern_size)
{
  kernel = new Eigen::MatrixXd (kern_size, kern_size);
  for (int i = 0; i < kern_size*kern_size; i++) {
    (*kernel)((int)i / kern_size,i%kern_size) = rand()/RAND_MAX;
  }
  output = new Eigen::MatrixXd (kern_size, kern_size); // We're using valid padding for now.
  for (int i = 0; i < kern_size*kern_size; i++) {
    (*output)((int)i / kern_size,i%kern_size) = rand()/RAND_MAX;
  }
};

void ConvLayer::convolute()
{
  for (int i = 0; i < input->cols() - kernel->cols() + 1; i+=stride_len) {
    for (int j = 0; j < input->rows() - kernel->rows() + 1; j+=stride_len) {
      (*output)(j, i) = (*kernel * (input->block(j, i, kernel->rows(), kernel->cols()))).sum();
    }
  }
}

class PoolingLayer
{
public:
  int stride_len;
  Eigen::MatrixXd* input;
  Eigen::MatrixXd* kernel;
  Eigen::MatrixXd* output;
  
  void pool();
  PoolingLayer(int x, int y, int stride, int kern_size);
};

// Will eventually be different from ConvLayer
PoolingLayer::PoolingLayer(int x, int y, int stride, int kern_size)
{
  kernel = new Eigen::MatrixXd (kern_size, kern_size);
  for (int i = 0; i < kern_size*kern_size; i++) {
    (*kernel)((int)i / kern_size,i%kern_size) = rand()/RAND_MAX;
  }
  output = new Eigen::MatrixXd (kern_size, kern_size);
  for (int i = 0; i < kern_size*kern_size; i++) {
    (*output)((int)i / kern_size,i%kern_size) = rand()/RAND_MAX;
  }
};

void PoolingLayer::pool()
{
  // It doesn't look like anything better than O(n^4) is doable for this as kernel needs to go through matrix and you need to index kernel. LOOK INTO ME!! 
  float maxnum = -LARGE_NUM;
  for (int i = 0; i < input->cols() - kernel->cols() + 1; i+=stride_len) {
    for (int j = 0; j < input->rows() - kernel->rows() + 1; j+=stride_len) {
      for (int k = 0; k < kernel->cols(); k++) {
        for (int l = 0; l < kernel->rows(); l++) {
          if ((input->block(j, i, kernel->rows(), kernel->cols()))(l, k) > maxnum) {
            maxnum = (input->block(j, i, kernel->rows(), kernel->cols()))(l, k);
          }
        }
      }
    }
  }
}

class ConvNet : public Network
{
public:
  int stride_len;
  int preprocess_length;
                
  std::vector<ConvLayer> conv_layers;
  std::vector<PoolingLayer> pool_layers;

  ConvNet(char* path, int batch_sz, float learn_rate, float bias_rate, float ratio);
  void process(); // Runs the convolutional and pooling layers.
  void next_batch();
  void add_conv_layer(int x, int y, int stride, int kern_size);
  void add_pool_layer(int x, int y, int stride, int kern_size);
};

ConvNet::ConvNet(char* path, int batch_sz, float learn_rate, float bias_rate, float ratio) : Network(path, batch_sz, learn_rate, bias_rate, ratio)
{
}

void ConvNet::add_conv_layer(int x, int y, int stride, int kern_size)
{
  preprocess_length++;
  conv_layers.emplace_back(x,y,stride,kern_size);
}

// May make this inaccessible to user code and just have it called from add_conv_layer as pooling is basically always paired with conv.
void ConvNet::add_pool_layer(int x, int y, int stride, int kern_size)
{
  preprocess_length++;
  pool_layers.emplace_back(x,y,stride,kern_size);
}

// Needs a batch advancement function, 100% does not work.
void ConvNet::process()
{
  // Assumes pooling is immediately after any conv layer.
  for (int i = 0; i < preprocess_length-1; i++) {
    conv_layers[i].convolute();
    pool_layers[i].input = conv_layers[i].output;
    pool_layers[i].pool();
    conv_layers[i+1].input = pool_layers[i].output;
  }
  conv_layers[preprocess_length-1].convolute();
  pool_layers[preprocess_length-1].input = conv_layers[preprocess_length-1].output;
  pool_layers[preprocess_length-1].pool();
  std::cout << "Output" << *pool_layers[preprocess_length-1].output << "\n\n";
  Eigen::Map<Eigen::RowVectorXd> flattened (pool_layers[preprocess_length-1].output->data(), pool_layers[preprocess_length-1].output->size());
  std::cout << "Flattened" << flattened << "\n\n";
  for (int i = 0; i < flattened.cols(); i++) {
    (*layers[0].contents)(0, i) = flattened[i];
  }
}

int main()
{
  ConvNet net ("./data_banknote_authentication.txt", 1, 0.05, 0.01, 0.9);
  net.add_conv_layer(8,8,1,4);
  net.add_pool_layer(4,4,1,2);
  net.add_layer(4, "linear");
  net.add_layer(5, "relu");
  net.add_layer(1, "resig");
  net.initialize();
  Eigen::MatrixXd* input = new Eigen::MatrixXd (8,8);
  *input << 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,0,0,
    0,0,1,1,1,1,0,0,
    0,0,1,1,1,1,0,0,
    0,0,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0;    
  net.conv_layers[0].input = input;
  net.process();
  net.list_net();
}
