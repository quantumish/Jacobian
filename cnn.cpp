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
  stride_len = stride;
  kernel = new Eigen::MatrixXd (kern_size, kern_size);
  for (int i = 0; i < kern_size*kern_size; i++) {
    (*kernel)((int)i / kern_size,i%kern_size) = (double) rand() / RAND_MAX;
  }
  output = new Eigen::MatrixXd (kern_size, kern_size); // We're using valid padding for now.
  for (int i = 0; i < kern_size*kern_size; i++) {
    (*output)((int)i / kern_size,i%kern_size) = (double) rand()/RAND_MAX;
  }
};

void ConvLayer::convolute()
{
  //std::cout << input->cols() << " " << input->cols() << "\n";
  //  std::cout << "Conv input:\n" << *input << "\nkernel:\n" << *kernel << "\n\n";
  for (int i = 0; i < input->cols() - kernel->cols(); i+=stride_len) {
    for (int j = 0; j < input->rows() - kernel->rows(); j+=stride_len) {
      //std::cout << i << j << stride_len << "\n";
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
  stride_len = stride;
  kernel = new Eigen::MatrixXd (kern_size, kern_size);
  for (int i = 0; i < kern_size*kern_size; i++) {
    (*kernel)((int)i / kern_size,i%kern_size) = (double) rand()/RAND_MAX;
  }
  output = new Eigen::MatrixXd (kern_size, kern_size);
  for (int i = 0; i < kern_size*kern_size; i++) {
    (*output)((int)i / kern_size,i%kern_size) = (double) rand()/RAND_MAX;
  }
};

void PoolingLayer::pool()
{
  // It doesn't look like anything better than O(n^4) is doable for this as kernel needs to go through matrix and you need to index kernel. LOOK INTO ME!! 
  float maxnum = -LARGE_NUM;
  //  std::cout << "Pool input:\n" << *input << "\n\n";
  for (int i = 0; i < input->cols() - kernel->cols(); i+=stride_len) {
    for (int j = 0; j < input->rows() - kernel->rows(); j+=stride_len) {
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
  void list_net();
  void process(); // Runs the convolutional and pooling layers.
  void next_batch();
  void add_conv_layer(int x, int y, int stride, int kern_size);
  void add_pool_layer(int x, int y, int stride, int kern_size);
};

ConvNet::ConvNet(char* path, int batch_sz, float learn_rate, float bias_rate, float ratio) : Network(path, batch_sz, learn_rate, bias_rate, ratio)
{
  preprocess_length = 0;
}

void ConvNet::add_conv_layer(int x, int y, int stride, int kern_size)
{
  preprocess_length+=1;
  conv_layers.emplace_back(x,y,stride,kern_size);
}

// May make this inaccessible to user code and just have it called from add_conv_layer as pooling is basically always paired with conv.
void ConvNet::add_pool_layer(int x, int y, int stride, int kern_size)
{
  pool_layers.emplace_back(x,y,stride,kern_size);
}

// Needs a batch advancement function, 100% does not work.
void ConvNet::process()
{
  //  std::cout << preprocess_length << "\n";
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
  //  std::cout << "Output:\n" << *pool_layers[preprocess_length-1].output << "\n\n";
  Eigen::Map<Eigen::RowVectorXd> flattened (pool_layers[preprocess_length-1].output->data(), pool_layers[preprocess_length-1].output->size());
  // std::cout << "Flattened:\n" << flattened << "\n\n";
  for (int i = 0; i < flattened.cols(); i++) {
    (*layers[0].contents)(0, i) = flattened[i];
  }
}

void ConvNet::list_net()
{
  for (int i = 0; i < preprocess_length; i++) {
    std::cout << " CONVOLUTIONAL LAYER " << i << "\n\n" << *conv_layers[i].input << "\n\n WITH KERNEL\n" << *conv_layers[i].kernel << "\n\n AND OUTPUT \n" << *conv_layers[i].output << "\n\n\n";
    std::cout << " POOLING LAYER " << i << "\n\n" << *pool_layers[i].input << "\n\n WITH KERNEL\n" << *pool_layers[i].kernel << "\n\n AND OUTPUT \n" << *pool_layers[i].output << "\n\n\n";
  }
  std::cout << " INPUT LAYER (LAYER 0)\n\n" << *layers[0].contents << "\n\n WITH BIAS\n" << *layers[0].bias << "\n\n AND WEIGHTS \n" << *layers[0].weights << "\n\n\n";
  for (int i = 1; i < length-1; i++) {
    std::cout << " LAYER " << i << "\n\n" << *layers[i].contents << "\n\n WITH BIAS\n" << *layers[i].bias << "\n\n AND WEIGHTS \n" << *layers[i].weights << "\n\n\n";
  }
  std::cout << " OUTPUT LAYER (LAYER " << length-1 << ")\n\n" << *layers[length-1].contents << "\n\n WITH BIAS\n" << *layers[length-1].bias <<  "\n\n\n";
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
  *input <<
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,0,0,
    0,0,1,1,1,1,0,0,
    0,0,1,1,1,1,0,0,
    0,0,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0;    
  net.conv_layers[0].input = input;
  net.process();
  net.feedforward();
  net.list_net();
}
