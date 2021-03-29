#ifndef BPNN_H
#define BPNN_H

#include <eigen3/Eigen/Dense>

#include <vector>
#include <iostream>
#include <string>
#include <cstdio>
#include <cmath>
#include <random>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <lz4.h>

#define BUFFER_SIZE 600*1024
#define LARGE_BUF 600*1024*15
enum Regularization {L1, L2};

class Layer {
public:
    Eigen::MatrixXf* contents;
    Eigen::MatrixXf* v;
    Eigen::MatrixXf* m;
    Eigen::MatrixXf* weights;
    Eigen::MatrixXf* bias;
    Eigen::MatrixXf* dZ;
    std::function<float(float)> activation;
    std::function<float(float)> activation_deriv;
    char activation_str[32];

    Layer(int rows, int columns);
    Layer(float* vals, int rows, int columns);
    void operator=(const Layer& that);
    void init_weights(Layer next);
};

class Network {
    char buf[BUFFER_SIZE];
    char* p;
protected:
    int instances;
    float epoch_acc;
    float epoch_cost;
    float val_acc;
    float val_cost;
    std::function<void(void)> decay;
    std::function<void(std::vector<Eigen::MatrixXf>, int, int)> grad_calc;
    std::function<void(Layer&, Eigen::MatrixXf, float)> update;
    void next_batch(int fd);
public:
    int data;
    int val_data;
    int val_instances;
    int test_instances;
    std::vector<Layer> layers;
    int length = 0;
    int batch_size;
    float learning_rate;
    float bias_lr;
    Regularization reg_type;
    float lambda;
    bool early_stop;
    float threshold;
    bool silenced = false;
    int epochs = 0;
    int batches = 0;
    Eigen::MatrixXf* labels;

    Network(const char* path, int batch_sz, float learn_rate,
            float bias_rate, Regularization regularization,
            float l, float ratio, bool early_exit=true, float cutoff=0);
    ~Network();
    void add_layer(int nodes, std::function<float(float)> activation, std::function<float(float)> activation_deriv);
    void initialize();
    void init_optimizer(std::function<void(Layer&, Eigen::MatrixXf, float)> f);
    void set_activation(int index, std::function<float(float)> custom, std::function<float(float)> custom_deriv);
    void feedforward();
    void softmax();
    void list_net();
    float cost();
    float accuracy();
    Eigen::MatrixXf backpropagate();
    void validate(const char* path);
    void train();
    float get_acc() {return epoch_acc;}
    float get_val_acc() {return val_acc;}
    float get_cost() {return epoch_cost;}
    float get_val_cost() {return val_cost;}
};

int prep_file(const char* path, const char* out_path);
int split_file(const char* path, int lines, float ratio);

void prep(const char* rname, const char* wname);
void compress(const char* rname, const char* wname);
Eigen::MatrixXf l1_deriv(Eigen::MatrixXf m);

namespace optimizers {
std::function<void(Layer&, Eigen::MatrixXf, float)> momentum(float beta);
std::function<void(Layer&, Eigen::MatrixXf, float)> demon(float beta_init, int max_ep);
std::function<void(Layer&, Eigen::MatrixXf, float)> adam(float beta1, float beta2, float epsilon);
std::function<void(Layer&, Eigen::MatrixXf, float)> adamax(float beta1, float beta2, float epsilon);
}

#define MAXLINE 1024

#if (!RECKLESS)
#define checknan(x, loc) if(x==INFINITY || x==NAN || x == -INFINITY) throw ValueError("Detected NaN in operation", loc)
#define Expects(cond) assert(cond);
#define Ensures(cond) assert(cond);
#else
#define checknan(x, loc)
#define Expects(cond) GSL_ASSUME(cond);
#define Ensures(cond) GSL_ASSUME(cond);
#endif

#define SHUFFLED_PATH "./shuffled.txt"
#define VAL_PATH "./test.txt"
#define TRAIN_PATH "./train.txt"
#define VAL_BIN_PATH "./test.bin"
#define TRAIN_BIN_PATH "./train.bin"
#define VAL_LZ4_PATH "./test.lz4"
#define TRAIN_LZ4_PATH "./train.lz4"

#endif /* MODULE_H */
