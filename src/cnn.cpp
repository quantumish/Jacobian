//
//  cnn.cpp
//  Jacobian
//
//  Created by David Freifeld
//

#include "bpnn.hpp"
#include "utils.hpp"

//#include <Eigen/unsupported/CXX11/Tensor>
#include "cnn.hpp"

#define LARGE_NUM 1000000 // Remove me.

#if (!RECKLESS)
#define checknan(x, loc) if(x==INFINITY || x==NAN || x == -INFINITY) throw ValueError("Detected NaN in operation", loc)
#else
#define checknan(x, loc)
#endif

// NOTE: Below three functions not mine, from https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set/
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

ConvLayer::ConvLayer(int x, int y, int stride, int kern_x, int kern_y, int pad, std::function<float(float)> activ, std::function<float(float)> activ_deriv)
    :padding(pad), stride_len(stride), activation(activ), activation_deriv(activ_deriv)
{
    pad*=2;
    input = new Eigen::MatrixXf (x+pad,y+pad);
    dZ = new Eigen::MatrixXf (x+pad,y+pad);
    for (int i = 0; i < (x+pad)*(y+pad); i++) {
        (*input)((int)i / (y+pad),i%(y+pad)) = 0;
        (*dZ)((int)i / (y+pad),i%(y+pad)) = 0;        
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
    for (int i = 0; i < (*input).rows(); i++) {
        for (int j = 0; j < (*input).cols(); j++) {
            (*dZ)(i, j) = activation_deriv((*input)(i, j));
            (*input)(i, j) = activation((*input)(i, j));
        }
    }
    for (int i = 0; i < output->rows(); i+=stride_len) {
        for (int j = 0; j < output->cols(); j+=stride_len) {
            (*output)(i, j) = (*kernel * (input->block(i, j, kernel->rows(), kernel->cols()))).sum();            
        }
    }
    *output = (output->array() + bias).matrix();
}

void ConvLayer::set_input(Eigen::MatrixXf* matrix)
{
    input->block(padding, padding, matrix->rows(), matrix->cols()) = *matrix;
}

// Will eventually be different from ConvLayer
PoolingLayer::PoolingLayer(int x, int y, int stride, int kern_x, int kern_y, int pad)
  :padding(pad), stride_len(stride)
{
    input = new Eigen::MatrixXf (x+pad,y+pad);
    for (int i = 0; i < (x+pad)*(y+pad); i++) {
        (*input)((int)i / (y+pad),i%(y+pad)) = 0;
    }
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

ConvNet::ConvNet(char* path, float learn_rate, float bias_rate, Regularization reg, float l, float ratio)
    :Network(path, 1, learn_rate, bias_rate, reg, l, ratio), preprocess_length{0}
{
    ReadMNIST(10000,784,data);
    data_labels = read_mnist_labels("./t10k-labels-idx1-ubyte",10000);
    labels = new Eigen::MatrixXf (1, 1);
}

void ConvNet::add_conv_layer(int x, int y, int stride, int kern_x, int kern_y, int pad, std::function<float(float)> activ, std::function<float(float)> activ_deriv)
{
    preprocess_length+=1;
    conv_layers.emplace_back(x,y,stride,kern_x, kern_y,pad,activ,activ_deriv);
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
    // Assumes pooling is immediately after any conv layer.
    for (int i = 0; i < preprocess_length-1; i++) {
        conv_layers[i].convolute();
        //   pool_layers[i].input = conv_layers[i].output;
        //   pool_layers[i].pool();
        conv_layers[i+1].input = conv_layers[i].output;
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
    for (int i = 0; i < error.rows(); i++) {
        for (int j = 0; j < error.cols(); j++) {
            float truth;
            if (j==(*labels)(i,0)) truth = 1;
            else truth = 0;
            error(i,j) = (*layers[length-1].contents)(i,j) - truth;
            checknan(error(i,j), "gradient of final layer");
        }
    }
    gradients.push_back(error);
    deltas.push_back((*layers[length-2].contents).transpose() * gradients[0]);
    int counter = 1;
    for (int i = length-2; i >= 1; i--) {
        gradients.push_back((gradients[counter-1] * layers[i].weights->transpose()).cwiseProduct(*layers[i].dZ));
        deltas.push_back(layers[i-1].contents->transpose() * gradients[counter]);
        counter++;
    }
    for (int i = 0; i < length-1; i++) {
        update(deltas, i);
        if (reg_type == 2) *layers[length-2-i].weights -= ((lambda/batch_size) * (*layers[length-2-i].weights));
        else if (reg_type == 1) *layers[length-2-i].weights -= ((lambda/(2*batch_size)) * l1_deriv(*layers[length-2-i].weights));
        *layers[length-1-i].bias -= bias_lr * gradients[i];
    }
    Eigen::Map<Eigen::MatrixXf> reshaped(gradients[gradients.size()-1].data(),
                           conv_layers.back().output->rows(),
                           conv_layers.back().output->cols());
    gradients[gradients.size()-1] = reshaped;
    std::vector<Eigen::MatrixXf> conv_deltas;
    for (int layer = conv_layers.size()-1; layer >= 0; layer--) {
        conv_deltas.emplace_back(conv_layers[layer].kernel->rows(),
                                 conv_layers[layer].kernel->cols());
        std::cout << conv_layers[layer].input->cols() << " " << gradients.back().cols() << "\n";
        for (int i = 0; i < conv_layers[layer].input->rows() - gradients.back().rows() + 1; i++) {
            for (int j = 0; j < conv_layers[layer].input->cols() - gradients.back().cols() + 1; j++) {
                conv_deltas[conv_deltas.size()-1](i, j) = (gradients.back() * conv_layers[layer].input->block(i, j, gradients.back().rows(), gradients.back().cols())).sum();
            }
        }
        *conv_layers[layer].kernel -= conv_deltas.back();

        Eigen::MatrixXf flipped_kernel =
            Eigen::MatrixXf::Zero(conv_layers[layer].kernel->rows(), conv_layers[layer].kernel->cols());
        flipped_kernel = conv_layers[layer].kernel->transpose().colwise().reverse().transpose().colwise().reverse();
        Eigen::MatrixXf padded_grad = Eigen::MatrixXf::Zero(gradients.back().rows() + ((flipped_kernel.rows() - 1)*2), gradients.back().cols() + ((flipped_kernel.cols() - 1)*2));
        padded_grad.block(flipped_kernel.rows() - 1, flipped_kernel.cols() - 1, gradients.back().rows(), gradients.back().cols()) = gradients.back();
        Eigen::MatrixXf final_grad (padded_grad.rows() - flipped_kernel.rows() + 1, padded_grad.cols() - flipped_kernel.cols() + 1);
        for (int i = 0; i < padded_grad.rows() - flipped_kernel.rows() + 1; i++) {
            for (int j = 0; j < padded_grad.cols() - gradients.back().cols() + 1; j++) {
                final_grad(i, j) = (flipped_kernel * padded_grad.block(i, j, flipped_kernel.rows(), flipped_kernel.cols())).sum();
            }
        }
        gradients.push_back(final_grad.cwiseProduct(*conv_layers[layer].dZ));
    }
}

void ConvNet::train()
{
    float cost_sum = 0;
    float acc_sum = 0;
    for (int i = 0; i <= 100; i++) {
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
    list_net();
    epoch_acc = 1.0/(100) * acc_sum;
    epoch_cost = 1.0/(100) * cost_sum;
    printf("Epoch %i complete - cost %f - acc %f\n", epochs, epoch_cost, epoch_acc);
    batches=0;
    decay();
    epochs++;
}

int main()
{
    ConvNet net ("../data_banknote_authentication.txt", 0.05, 0.01, L2, 0, 0.9);
    Eigen::MatrixXf labels (1,1);
    net.add_conv_layer(28, 28, 1, 9, 9, 0, lecun_tanh, lecun_tanh_deriv);
    //  net.add_pool_layer(20,20,1,6,6,0);
    net.add_conv_layer(20, 20, 1, 6, 6, 0, lecun_tanh, lecun_tanh_deriv);
    std::cout << net.conv_layers[net.conv_layers.size()-1].output->rows() << "\n";
    //net.add_pool_layer(10,10,1,2,2,0);
    net.add_layer(400, "sigmoid", sigmoid, sigmoid_deriv);
    net.add_layer(5, "lecun_tanh", lecun_tanh, lecun_tanh_deriv);
    net.add_layer(10, "resig", rectifier(sigmoid), rectifier(sigmoid_deriv));
    //  net.list_net();
    //  net.init_decay("step", 1, 2);
    net.initialize();
    //net.list_net();

    for (int i = 0; i < 1; i++) {
        net.train();
    }
    net.list_net();
}
