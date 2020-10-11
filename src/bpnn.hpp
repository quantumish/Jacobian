#ifndef BPNN_H
#define BPNN_H

#include <Eigen/Dense>

//#include "../../mapreduce/mapreduce.h"

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
#include <gsl/gsl_assert>
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
    char activation_str[1024];
    // TODO: Fix PReLU layer inheritance | -p C -m PReLU layers shouldn't be Layers but inherit from them!
    float alpha;
  
    Layer(int rows, int columns, float a=0);
    Layer(float* vals, int rows, int columns);
    void operator=(const Layer& that);
    void init_weights(Layer next);
};

class Network {
public:
    int data;
    int val_data;
    int instances;
    char buf[LARGE_BUF];
    char* p;
    int read_len;
    int val_instances;
    int test_instances;
    Eigen::MatrixXf numerical_grad(int i, float epsilon);
    void update_layer(float* vals, int datalen, int index);
  
    std::vector<Layer> layers;
    int length = 0;

    float epoch_acc;
    float epoch_cost;
    float val_acc;
    float val_cost;
  
    float learning_rate;
    float bias_lr;
    float lambda;
    Regularization reg_type;
    int batch_size;

    bool silenced = false;
    int epochs = 0;
    int batches = 0;
    Eigen::MatrixXf* labels;

    std::function<void(void)> decay;
    std::function<void(std::vector<Eigen::MatrixXf>, int, int)> grad_calc;
    std::function<void(std::vector<Eigen::MatrixXf>, int)> update;

    Network(char* path, int batch_sz, float learn_rate,
            float bias_rate, Regularization regularization,
            float l, float ratio, bool early_exit=true, float cutoff=0);
    
    void add_layer(int nodes, char* name, std::function<float(float)> activation, std::function<float(float)> activation_deriv);
    void add_prelu_layer(int nodes, float a);
    void init_decay(char* type, ...);
    void init_optimizer(char* name, ...);
    void initialize();
    void grad_check();
    void set_activation(int index, std::function<float(float)> custom, std::function<float(float)> custom_deriv);
  
    void feedforward();
    void softmax();
    void list_net();

    bool early_stop;
    float threshold;

    float cost();
    float accuracy();
    void backpropagate();
    int next_batch(int fd);
    float validate(char* path);
    void train();

    float get_acc() {return epoch_acc;}
    float get_val_acc() {return val_acc;}
    float get_cost() {return epoch_cost;}
    float get_val_cost() {return val_cost;}
};

int prep_file(char* path, char* out_path);
int split_file(char* path, int lines, float ratio);

struct ValueError : public std::exception
{
    const char* message;
    const char* location;
    ValueError(const char* msg, const char* loc)
        :message{msg}, location{loc}
    {
    }
    const char* what() const throw () {
        char* error;
        sprintf(error, "%s (thrown in %s).", message, location);
        const char* error_message = error;
        return error_message;
    }
};
void prep(char* rname, char* wname);
void compress(char* rname, char* wname);

#define MAXLINE 1024

#if (!RECKLESS)
#define checknan(x, loc) if(x==INFINITY || x==NAN || x == -INFINITY) throw ValueError("Detected NaN in operation", loc)
#else
#define checknan(x, loc)
#define Expects(cond) GSL_ASSUME(cond);
#define Ensures(cond) GSL_ASSUME(cond);
#endif

#endif /* MODULE_H */
