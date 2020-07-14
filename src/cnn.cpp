#include "bpnn.hpp"
#include "utils.hpp"

#define LARGE_NUM 1000000 // Remove me.

#if (!RECKLESS)
#define checknan(x, loc) if(x==INFINITY || x==NAN || x == -INFINITY) throw ValueError("Detected NaN in operation", loc)
#else
#define checknan(x, loc)
#endif

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}   

void ReadMNIST(int NumberOfImages, int DataOfAnImage,std::vector<std::vector<double>> &arr)
{
    arr.resize(NumberOfImages,std::vector<double>(DataOfAnImage));
    std::ifstream file ("./t10k-images-idx3-ubyte",std::ios::binary);
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

unsigned char* read_mnist_labels(std::string full_path, int number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        uchar* _dataset = new uchar[number_of_labels];
        for(int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
        return _dataset;
    } else {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}

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
  std::vector<std::vector<double>> data;
  unsigned char* data_labels;
                
  std::vector<ConvLayer> conv_layers;
  std::vector<PoolingLayer> pool_layers;
  
  ConvNet(char* path, float learn_rate, float bias_rate, float l, float ratio);
  void list_net();
  void process(); // Runs the convolutional and pooling layers.
  void next_batch();
  void backpropagate();
  void train();
  void add_conv_layer(int x, int y, int stride, int kern_x, int kern_y, int pad);
  void add_pool_layer(int x, int y, int stride, int kern_x, int kern_y, int pad);
  void set_label(Eigen::MatrixXf newlabels);
  void initialize();
};

ConvNet::ConvNet(char* path, float learn_rate, float bias_rate, float l, float ratio)
  : Network(path, 1, learn_rate, bias_rate, l, ratio), preprocess_length{0}
{
  ReadMNIST(10000,784,data);
  data_labels = read_mnist_labels("./t10k-labels-idx1-ubyte",10000);
  labels = new Eigen::MatrixXf (1, 1);
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

void ConvNet::next_batch()
{
  for (int i = 0; i < 784; i++) {
    (*conv_layers[0].input)(i/28, i%28) = data[batches][i];
  }
  (*labels)(0,0) = (float)(int)data_labels[batches];
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
  Eigen::MatrixXf error (layers[length-1].contents->rows(), layers[length-1].contents->cols());
  // std::cout << (*layers[length-1].contents) << "\n\n\n";
  for (int i = 0; i < error.rows(); i++) {
    for (int j = 0; j < error.cols(); j++) {
      float truth;
      if (j==(*labels)(i,0)) truth = 1;
      else truth = 0;
      error(i,j) = (*layers[length-1].contents)(i,j) - truth;
      //      std::cout << error(i, j) << " " << (*layers[length-1].contents)(i,j) << " " << truth << "\n";
      checknan(error(i, j), "gradient of final layer");
      // std::cout << truth << "[as label is "<< (*labels)(i,0) <<"] - " << (*layers[length-1].contents)(i,j) << "[aka index " << i << " " << j << "] = " << error(i,j) << "\n";
    }
  }
  gradients.push_back(error);
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
    std::cout << learning_rate << " (LR) \n" << deltas[i] << "\n\n";
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

void ConvNet::train()
{
  float cost_sum = 0;
  float acc_sum = 0;
  for (int i = 0; i <= 1; i++) {
    if (i != instances-batch_size) { // Don't try to advance batch on final batch.
      next_batch();
    }
    process();
    feedforward();
    backpropagate();
    cost_sum += cost();
    acc_sum += accuracy();
    batches++;
  }
  epoch_acc = 1.0/(10000) * acc_sum;
  epoch_cost = 1.0/(10000) * cost_sum;
  printf("Epoch %i complete - cost %f - acc %f\n", epochs, epoch_cost, epoch_acc);
  batches=0;
  learning_rate = decay(learning_rate, epochs);
  epochs++;
}

int main()
{
  ConvNet net ("../data_banknote_authentication.txt", 0.05, 0.01, 5, 0.9);
  Eigen::MatrixXf labels (1,1);
  labels << 2;
  net.set_label(labels);
  net.add_conv_layer(28,28,1,4,4,0);
  //net.add_pool_layer(5,5,1,2,0);
  net.add_layer(625, "linear");
  net.add_layer(5, "relu");
  net.add_layer(10, "linear");
  net.init_decay("step", 1, 2);
  net.initialize();

  for (int i = 0; i < 1; i++) {
    net.train();
  }
  std::cout << *net.layers[net.length-1].contents << "\n";
}
