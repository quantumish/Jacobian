#ifndef CNN_H
#define CNN_H

#include <fstream>

class ConvLayer
{
public:
    int stride_len;
    int padding;
    Eigen::MatrixXf* input;
    Eigen::MatrixXf* kernel;
    Eigen::MatrixXf* output;
    Eigen::MatrixXf* dZ;
    std::function<float(float)> activation;
    std::function<float(float)> activation_deriv;
    float bias;
  
    ConvLayer(int x, int y, int stride, int kern_x, int kern_y, int pad, std::function<float(float)> activ, std::function<float(float)> activ_deriv);
    void convolute();
    void set_input(Eigen::MatrixXf* matrix);
};

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

class ConvNet : public Network
{
public:
    int preprocess_length;
    std::vector<std::vector<double>> data;
    unsigned char* data_labels;
    
    std::vector<ConvLayer> conv_layers;
    std::vector<PoolingLayer> pool_layers;
    
    ConvNet(char* path, float learn_rate, float bias_rate, Regularization reg, float l, float ratio);
    void list_net();
    void process(); // Runs the convolutional and pooling layers.
    void next_batch();
    void backpropagate();
    void train();
    void add_conv_layer(int x, int y, int stride, int kern_x, int kern_y, int pad, std::function<float(float)> activ, std::function<float(float)> activ_deriv);
    void add_pool_layer(int x, int y, int stride, int kern_x, int kern_y, int pad);
    void set_label(Eigen::MatrixXf newlabels);
    void initialize();
};
#endif /* MODULE_H */
