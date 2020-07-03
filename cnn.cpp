#include "bpnn.hpp"
#include "utils.hpp"

#define LARGE_NUM 1000000 // Remove me.

class ConvLayer
{
public:
  int stride_len;
  int padding;
  Eigen::MatrixXd* input;
  Eigen::MatrixXd* kernel;
  Eigen::MatrixXd* output;
  Eigen::MatrixXd* bias;
  
  ConvLayer(int x, int y, int stride, int kernel_size, int pad);
  void convolute();
};

ConvLayer::ConvLayer(int x, int y, int stride, int kern_size, int pad)
{
  padding = pad;
  stride_len = stride;
  kernel = new Eigen::MatrixXd (kern_size, kern_size);
  for (int i = 0; i < kern_size*kern_size; i++) {
    (*kernel)((int)i / kern_size,i%kern_size) = (double) rand() / RAND_MAX;
  }
  output = new Eigen::MatrixXd (x-kern_size+1, y-kern_size+1); // We're using valid padding for now.
  for (int i = 0; i < (x-kern_size+1)*(y-kern_size+1); i++) {
    (*output)((int)i / (y-kern_size+1),i%(y-kern_size+1)) = (double) rand()/RAND_MAX;
  }
};

void ConvLayer::convolute()
{
  for (int i = 0; i < input->cols() - kernel->cols()+1; i+=stride_len) {
    for (int j = 0; j < input->rows() - kernel->rows()+1; j+=stride_len) {
      (*output)(j, i) = (*kernel * (input->block(j, i, kernel->rows(), kernel->cols()))).sum();
    }
  }
}

class PoolingLayer
{
public:
  int stride_len;
  int padding;
  Eigen::MatrixXd* input;
  Eigen::MatrixXd* kernel;
  Eigen::MatrixXd* output;
  
  void pool();
  PoolingLayer(int x, int y, int stride, int kern_size, int pad);
};

// Will eventually be different from ConvLayer
PoolingLayer::PoolingLayer(int x, int y, int stride, int kern_size, int pad)
{
  padding = pad;
  stride_len = stride;
  kernel = new Eigen::MatrixXd (kern_size, kern_size);
  for (int i = 0; i < kern_size*kern_size; i++) {
    (*kernel)((int)i / kern_size,i%kern_size) = (double) rand()/RAND_MAX;
  }
  output = new Eigen::MatrixXd (x-kern_size+1, y-kern_size+1);
  for (int i = 0; i < (x-kern_size+1)*(y-kern_size+1); i++) {
    (*output)((int)i / (y-kern_size+1),i%(y-kern_size+1)) = (double) rand()/RAND_MAX;
  }
};

void PoolingLayer::pool()
{
  // It doesn't look like anything better than O(n^4) is doable for this as kernel needs to go through matrix and you need to index kernel. LOOK INTO ME!! 
  float maxnum = -LARGE_NUM;
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
  int preprocess_length;
                
  std::vector<ConvLayer> conv_layers;
  std::vector<PoolingLayer> pool_layers;

  ConvNet(char* path, int batch_sz, float learn_rate, float bias_rate, float ratio);
  void list_net();
  void process(); // Runs the convolutional and pooling layers.
  void backpropagate();
  void next_batch();
  void add_conv_layer(int x, int y, int stride, int kern_size, int pad);
  void add_pool_layer(int x, int y, int stride, int kern_size, int pad);
};

ConvNet::ConvNet(char* path, int batch_sz, float learn_rate, float bias_rate, float ratio) : Network(path, batch_sz, learn_rate, bias_rate, ratio)
{
  preprocess_length = 0;
}

void ConvNet::add_conv_layer(int x, int y, int stride, int kern_size, int pad)
{
  preprocess_length+=1;
  conv_layers.emplace_back(x,y,stride,kern_size,pad);
}

// May make this inaccessible to user code and just have it called from add_conv_layer as pooling is basically always paired with conv.
void ConvNet::add_pool_layer(int x, int y, int stride, int kern_size, int pad)
{
  pool_layers.emplace_back(x,y,stride,kern_size,pad);
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
  //pool_layers[preprocess_length-1].input = conv_layers[preprocess_length-1].output;
  //pool_layers[preprocess_length-1].pool();
  //  std::cout << "Output:\n" << *pool_layers[preprocess_length-1].output << "\n\n";
  Eigen::Map<Eigen::RowVectorXd> flattened (conv_layers[preprocess_length-1].output->data(), conv_layers[preprocess_length-1].output->size());
  // std::cout << "Flattened:\n" << flattened << "\n\n";
  for (int i = 0; i < flattened.cols(); i++) {
    (*layers[0].contents)(0, i) = flattened[i];
  }
}

void ConvNet::list_net()
{
  for (int i = 0; i < preprocess_length; i++) {
    std::cout << "-----------------------\nCONVOLUTIONAL LAYER " << i << "\n-----------------------\n\n\u001b[31mGENERAL INFO:\x1B[0;37m\nStride: " << conv_layers[i].stride_len << "\nPadding: " << conv_layers[i].padding <<  "\n\n\u001b[31mINPUT:\x1B[0;37m\n" << *conv_layers[i].input << "\n\n\u001b[31mKERNEL:\x1B[0;37m\n" << *conv_layers[i].kernel << "\n\n\u001b[31mOUTPUT:\x1B[0;37m\n" << *conv_layers[i].output << "\n\n\n";
    std::cout << "-----------------------\nPOOLING LAYER " << i << "\n-----------------------\n\n\u001b[31mGENERAL INFO:\x1B[0;37m\nStride: " << pool_layers[i].stride_len << "\nPadding: " << conv_layers[i].padding << "\n\n\u001b[31mINPUT:\x1B[0;37m\n" << *pool_layers[i].input << "\n\n\u001b[31mKERNEL:\x1B[0;37m\n-" << *pool_layers[i].kernel << "\n\n\u001b[31mOUTPUT:\x1B[0;37m\n" << *pool_layers[i].output << "\n\n\n";
  }
  std::cout << "-----------------------\nINPUT LAYER (LAYER 0)\n-----------------------\n\n\u001b[31mGENERAL INFO:\x1B[0;37m\nActivation Function: " << layers[0].activation_str << "\n\n\u001b[31mACTIVATIONS:\x1B[0;37m\n" << *layers[0].contents << "\n\n\u001b[31mWEIGHTS:\x1B[0;37m\n" << *layers[0].weights << "\n\n\u001b[31mBIASES:\x1B[0;37m\n" << *layers[0].bias << "\n\n\n";
  for (int i = 1; i < length-1; i++) {
    std::cout << "-----------------------\nLAYER " << i << "\n-----------------------\n\n\u001b[31mGENERAL INFO:\x1B[0;37m\nActivation Function: " << layers[i].activation_str << "\n\n\u001b[31mACTIVATIONS:\x1B[0;37m\n" << *layers[i].contents << "\n\n\u001b[31mBIASES:\x1B[0;37m\n" << *layers[i].bias << "\n\n\u001b[31mWEIGHTS:\x1B[0;37m\n" << *layers[i].weights << "\n\n\n";
  }
  std::cout << "-----------------------\nOUTPUT LAYER (LAYER " << length-1 << ")\n-----------------------\n\n\u001b[31mGENERAL INFO:\x1B[0;37m\nActivation Function: " << layers[length-1].activation_str <<"\n\n\u001b[31mACTIVATIONS:\x1B[0;37m\n" << *layers[length-1].contents << "\n\n\u001b[31mBIASES:\x1B[0;37m\n" << *layers[length-1].bias <<  "\n\n\n";
}

void ConvNet::backpropagate()
{
  std::vector<Eigen::MatrixXd> gradients;
  std::vector<Eigen::MatrixXd> deltas;
  Eigen::MatrixXd error = ((*layers[length-1].contents) - (*labels));
  gradients.push_back(error.cwiseProduct(*layers[length-1].dZ));
  deltas.push_back((*layers[length-2].contents).transpose() * gradients[0]);
  int counter = 1;
  for (int i = length-2; i >= 1; i--) {
    gradients.push_back((gradients[counter-1] * layers[i].weights->transpose()).cwiseProduct(*layers[i].dZ));
    deltas.push_back(layers[i-1].contents->transpose() * gradients[counter]);
    counter++;
  }
  for (int i = 0; i < length-1; i++) {
    *layers[length-2-i].weights -= learning_rate * deltas[i];   
    *layers[length-1-i].bias -= bias_lr * gradients[i];
  }
  for (int i = 0; i < conv_layers[0].input->cols() - gradients[0].cols()+1; i+=conv_layers[0].stride_len) {
    for (int j = 0; j < conv_layers[0].input->rows() - gradients[0].rows()+1; j+=conv_layers[0].stride_len) {
      (*conv_layers[0].kernel)(j, i) -= (gradients[0] * (conv_layers[0].input->block(j, i, gradients[0].rows(), gradients[0].cols()))).sum();
    }
  }
  *conv_layers[0].kernel -= *conv_layers[0].input * gradients[0];
}

int main()
{
  ConvNet net ("./data_banknote_authentication.txt", 1, 0.05, 0.01, 0.9);
  net.add_conv_layer(8,8,1,4,0);
  //net.add_pool_layer(5,5,1,2,0);
  net.add_layer(25, "linear");
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
  net.backpropagate();
  net.list_net();
}
