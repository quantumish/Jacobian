#include "bpnn.hpp"
#include "utils.hpp"

#define LARGE_NUM 1000000 // Remove me.

class ConvLayer
{
public:
  int stride_len;
  int padding;
  Eigen::MatrixXf* input;
  Eigen::MatrixXf* kernel;
  Eigen::MatrixXf* output;
  float bias;
  
  ConvLayer(int x, int y, int stride, int kern_x, int kern_y, int pad);
  void convolute();
  void set_input(Eigen::MatrixXf* matrix);
};

ConvLayer::ConvLayer(int x, int y, int stride, int kern_x, int kern_y, int pad)
{
  padding = pad;
  pad*=2;
  stride_len = stride;
  input = new Eigen::MatrixXf (x+pad,y+pad);
  for (int i = 0; i < (x+pad)*(y+pad); i++) {
    (*input)((int)i / (y+pad),i%(y+pad)) = 0;
  }
  kernel = new Eigen::MatrixXf (kern_x, kern_y);
  for (int i = 0; i < kern_x*kern_y; i++) {
    (*kernel)((int)i / kern_y,i%kern_y) = (float) rand() / RAND_MAX;
  }
  output = new Eigen::MatrixXf ((x-kern_x+1+pad/stride_len), (y-kern_y+1+pad/stride_len));
  for (int i = 0; i < (x-kern_y+1+pad/stride_len)*(y-kern_x+1+pad/stride_len); i++) {
    (*output)((int)i / (y-kern_y+1+pad/stride_len),i%(y-kern_y+1+pad/stride_len)) = 0;
  }
  bias = 0;
};

void ConvLayer::convolute()
{
  for (int i = 0; i < input->cols() - kernel->cols()+1; i+=stride_len) {
    for (int j = 0; j < input->rows() - kernel->rows()+1; j+=stride_len) {
      (*output)(j, i) = (*kernel * (input->block(j, i, kernel->rows(), kernel->cols()))).sum();
    }
  }
  *output = (output->array() + bias).matrix();
}

void ConvLayer::set_input(Eigen::MatrixXf* matrix)
{
  input->block(padding, padding, matrix->rows(), matrix->cols()) = *matrix;
}

class PoolingLayer
{
public:
  int stride_len;
  int padding;
  Eigen::MatrixXf* input;
  Eigen::MatrixXf* kernel;
  Eigen::MatrixXf* output;
  
  void pool();
  PoolingLayer(int x, int y, int stride, int kern_x, int kern_y, int pad);
};

// Will eventually be different from ConvLayer
PoolingLayer::PoolingLayer(int x, int y, int stride, int kern_x, int kern_y, int pad)
{
  padding = pad;
  stride_len = stride;
  kernel = new Eigen::MatrixXf (kern_x, kern_y);
  for (int i = 0; i < kern_x*kern_y; i++) {
    (*kernel)((int)i / kern_y,i%kern_y) = (float) rand()/RAND_MAX;
  }
  output = new Eigen::MatrixXf (x-kern_x+1, y-kern_y+1);
  for (int i = 0; i < (x-kern_x+1)*(y-kern_y+1); i++) {
    (*output)((int)i / (y-kern_y+1),i%(y-kern_y+1)) = (float) rand()/RAND_MAX;
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
  
  ConvNet(char* path, int batch_sz, float learn_rate, float bias_rate, float l, float ratio);
  void list_net();
  void process(); // Runs the convolutional and pooling layers.
  void backpropagate();
  void add_conv_layer(int x, int y, int stride, int kern_x, int kern_y, int pad);
  void add_pool_layer(int x, int y, int stride, int kern_x, int kern_y, int pad);
  void set_label(Eigen::MatrixXf newlabels);
  void initialize();
};

ConvNet::ConvNet(char* path, int batch_sz, float learn_rate, float bias_rate, float l, float ratio) : Network(path, batch_sz, learn_rate, bias_rate, l, ratio)
{
  preprocess_length = 0;
  labels = new Eigen::MatrixXf (batch_sz, 1);
}

void ConvNet::add_conv_layer(int x, int y, int stride, int kern_x, int kern_y, int pad)
{
  preprocess_length+=1;
  conv_layers.emplace_back(x,y,stride,kern_x, kern_y,pad);
}

// May make this inaccessible to user code and just have it called from add_conv_layer as pooling is basically always paired with conv.
void ConvNet::add_pool_layer(int x, int y, int stride, int kern_x, int kern_y, int pad)
{
  pool_layers.emplace_back(x,y,stride,kern_x,kern_y,pad);
}

void ConvNet::initialize()
{
  for (int i = 0; i < length-1; i++) {
    layers[i].init_weights(layers[i+1]);
  }
}

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
  Eigen::Map<Eigen::RowVectorXf> flattened (conv_layers[preprocess_length-1].output->data(), conv_layers[preprocess_length-1].output->size());
  // std::cout << "Flattened:\n" << flattened << "\n\n";
  for (int i = 0; i < flattened.cols(); i++) {
    (*layers[0].contents)(0, i) = flattened[i];
  }
}

void ConvNet::set_label(Eigen::MatrixXf newlabels)
{
  *labels = newlabels;
}

void ConvNet::list_net()
{
  for (int i = 0; i < preprocess_length; i++) {
    std::cout << "-----------------------\nCONVOLUTIONAL LAYER " << i << "\n-----------------------\n\n\u001b[31mGENERAL INFO:\x1B[0;37m\nStride: " << conv_layers[i].stride_len << "\nPadding: " << conv_layers[i].padding <<  "\n\n\u001b[31mINPUT:\x1B[0;37m\n" << *conv_layers[i].input << "\n\n\u001b[31mKERNEL:\x1B[0;37m\n" << *conv_layers[i].kernel << "\n\n\u001b[31mOUTPUT:\x1B[0;37m\n" << *conv_layers[i].output << "\n\n\u001b[31mBIAS:\x1B[0;37m\n" << conv_layers[i].bias << "\n\n\n";
    //std::cout << "-----------------------\nPOOLING LAYER " << i << "\n-----------------------\n\n\u001b[31mGENERAL INFO:\x1B[0;37m\nStride: " << pool_layers[i].stride_len << "\nPadding: " << conv_layers[i].padding << "\n\n\u001b[31mINPUT:\x1B[0;37m\n" << *pool_layers[i].input << "\n\n\u001b[31mKERNEL:\x1B[0;37m\n-" << *pool_layers[i].kernel << "\n\n\u001b[31mOUTPUT:\x1B[0;37m\n" << *pool_layers[i].output << "\n\n\n";
  }
  std::cout << "-----------------------\nINPUT LAYER (LAYER 0)\n-----------------------\n\n\u001b[31mGENERAL INFO:\x1B[0;37m\nActivation Function: " << layers[0].activation_str << "\n\n\u001b[31mACTIVATIONS:\x1B[0;37m\n" << *layers[0].contents << "\n\n\u001b[31mWEIGHTS:\x1B[0;37m\n" << *layers[0].weights << "\n\n\u001b[31mBIASES:\x1B[0;37m\n" << *layers[0].bias << "\n\n\n";
  for (int i = 1; i < length-1; i++) {
    std::cout << "-----------------------\nLAYER " << i << "\n-----------------------\n\n\u001b[31mGENERAL INFO:\x1B[0;37m\nActivation Function: " << layers[i].activation_str << "\n\n\u001b[31mACTIVATIONS:\x1B[0;37m\n" << *layers[i].contents << "\n\n\u001b[31mBIASES:\x1B[0;37m\n" << *layers[i].bias << "\n\n\u001b[31mWEIGHTS:\x1B[0;37m\n" << *layers[i].weights << "\n\n\n";
  }
  std::cout << "-----------------------\nOUTPUT LAYER (LAYER " << length-1 << ")\n-----------------------\n\n\u001b[31mGENERAL INFO:\x1B[0;37m\nActivation Function: " << layers[length-1].activation_str <<"\n\n\u001b[31mACTIVATIONS:\x1B[0;37m\n" << *layers[length-1].contents << "\n\n\u001b[31mBIASES:\x1B[0;37m\n" << *layers[length-1].bias <<  "\n\n\n";
}

void ConvNet::backpropagate()
{
  std::vector<Eigen::MatrixXf> gradients;
  std::vector<Eigen::MatrixXf> deltas;
  Eigen::MatrixXf error = ((*layers[length-1].contents) - (*labels));
  gradients.push_back(error.cwiseProduct(*layers[length-1].dZ));
  deltas.push_back((*layers[length-2].contents).transpose() * gradients[0]);
  int counter = 1;
  for (int i = length-2; i >= 1; i--) {
    //std::cout << "--GRAD---\n" << gradients[counter-1] << "\n\n" << layers[i].weights->transpose() << "\n\n" << *layers[i].dZ << "\n\n";
    gradients.push_back((gradients[counter-1] * layers[i].weights->transpose()).cwiseProduct(*layers[i].dZ));
    //std::cout << "---DELTA---\n" << gradients[counter] << "\n\n" << layers[i].weights->transpose() << "\n\n" << *layers[i].dZ << "\n\n";
    deltas.push_back(layers[i-1].contents->transpose() * gradients[counter]);
    counter++;
  }
  gradients.push_back((gradients[gradients.size()-1] * layers[0].weights->transpose()).cwiseProduct(*layers[0].dZ));
  for (int i = 0; i < length-1; i++) {
    *layers[length-2-i].weights -= learning_rate * deltas[i];   
    *layers[length-1-i].bias -= bias_lr * gradients[i];
  }
  //  list_net();
  // std::cout << "GRADIENT LIST\n";
  // for (int i = 0; i < gradients.size(); i++) {
  //   std::cout  << gradients[i] << "\n\n";
  // }
  Eigen::Map<Eigen::MatrixXf> reshaped(gradients[gradients.size()-1].data(), conv_layers[conv_layers.size()-1].output->rows(),conv_layers[conv_layers.size()-1].output->cols());
  gradients[gradients.size()-1] = reshaped;
  //std::cout << gradients[gradients.size()-1].cols() << " " << conv_layers[0].input->cols() << " " << conv_layers[0].input->cols() - gradients[length-1].cols()+1 << "\n";
  for (int i = 0; i < conv_layers[0].input->cols() - gradients[length-1].cols()+1; i+=conv_layers[0].stride_len) {
    for (int j = 0; j < conv_layers[0].input->rows() - gradients[length-1].rows()+1; j+=conv_layers[0].stride_len) {
      (*conv_layers[0].kernel)(j, i) -= (gradients[length-1] * (conv_layers[0].input->block(j, i, gradients[length-1].rows(), gradients[length-1].cols()))).sum();
    }
  }
  conv_layers[0].bias -= gradients[gradients.size()-1].sum();
}

using namespace std;
int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}
void ReadMNIST(int NumberOfImages, int DataOfAnImage,vector<vector<double>> &arr)
{
    arr.resize(NumberOfImages,vector<double>(DataOfAnImage));
    ifstream file ("./t10k-images-idx3-ubyte",ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    arr[i][(n_rows*r)+c]= (double)temp;
                }
            }
        }
    }
}

int main()
{
  ConvNet net ("../data_banknote_authentication.txt", 1, 0.05, 0.01, 0, 0.9);
  Eigen::MatrixXf labels (1,1);
  labels << 1;
  net.set_label(labels);
  net.add_conv_layer(28,28,1,4,4,0);
  //net.add_pool_layer(5,5,1,2,0);
  net.add_layer(625, "linear");
  net.add_layer(5, "relu");
  net.add_layer(1, "resig");
  net.initialize();

  Eigen::MatrixXf* input = new Eigen::MatrixXf (28,28);
  vector<vector<double>> ar;
  ReadMNIST(10000,784,ar);
  for (int i = 0; i < 784; i++) {
    (*input)(i/28, i%28) = ar[1][i];
  }
  std::cout << "\n\n" << *input << "\n";
  
  net.conv_layers[0].set_input(input);
  net.process();
  for (int i = 0; i < 10; i++) {
    net.feedforward();
    net.backpropagate();
    printf("Epoch %i complete - cost %f - acc %f\n", net.epochs, net.cost(), net.accuracy());
  }
}
