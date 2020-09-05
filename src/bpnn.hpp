#ifndef BPNN_H
#define BPNN_H

#include <Eigen/Dense>

//#include "../../mapreduce/mapreduce.h"

#include <vector>
#include <array>
#include <iostream>
#include <string>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <random>
#include <algorithm>

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
    FILE* data;
    FILE* val_data;
    FILE* test_data;
    int instances;
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
    int reg_type;
    int batch_size;

    bool silenced = false;
    int epochs = 0;
    int batches = 0;
    Eigen::MatrixXf* labels;

    std::function<void(void)> decay;
    std::function<void(std::vector<Eigen::MatrixXf>, int, int)> grad_calc;
    std::function<void(std::vector<Eigen::MatrixXf>, int)> update;

    Network(char* path, int batch_sz, float learn_rate, float bias_rate, int regularization, float l, float ratio, bool early_exit=true, float cutoff=0);
    void add_layer(int nodes, char* activation);
    void add_prelu_layer(int nodes, float a);
    void init_decay(char* type, ...);
    void init_optimizer(char* name, ...);
    void initialize();
    void grad_check();
    void set_activation(int index, std::function<float(float)> custom, std::function<float(float)> custom_deriv);
  
    void feedforward();
    void list_net();

    bool early_stop;
    float threshold;

    float cost();
    float accuracy();
    void backpropagate();
    int next_batch();
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

#define MAXLINE 1024

#if (!RECKLESS)
#define checknan(x, loc) if(x==INFINITY || x==NAN || x == -INFINITY) throw ValueError("Detected NaN in operation", loc)
#else
#define checknan(x, loc)
#endif

#endif /* MODULE_H */
