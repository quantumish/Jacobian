#ifndef BPNN_H
#define BPNN_H

#include "/Users/davidfreifeld/Downloads/eigen-3.3.7/Eigen/Dense"

extern "C" {
  #include "../mapreduce/mapreduce.h"
}

#include <vector>
#include <array>
#include <iostream>
#include <string>
#include <cstdio>
#include <fstream>
#include <random>
#include <algorithm>

class Layer {
public:
  Eigen::MatrixXd* contents;
  Eigen::MatrixXd* weights;
  Eigen::MatrixXd* bias;
  Eigen::MatrixXd* dZ;
  std::function<double(double)> activation;
  std::function<double(double)> activation_deriv;
  
  Layer(int rows, int columns);
  Layer(float* vals, int rows, int columns);
  void init_weights(Layer next);
};

class Network {
public:
  FILE* data;
  int instances;

  std::vector<Layer> layers;
  int length;

  float learning_rate;
  float bias_lr;
  int batch_size;
  int batches;
  Eigen::MatrixXd* labels;

  Network(char* path, int batch_sz, float learn_rate, float bias_rate);
  void add_layer(int nodes, char* activation);
  void initialize();
  void update_layer(float* vals, int datalen, int index);
  void set_activation(int index, std::function<double(double)> custom, std::function<double(double)> custom_deriv);
  
  Eigen::MatrixXd init_ones(Eigen::MatrixXd matrix);
  void feedforward();
  void list_net();

  float cost();
  float accuracy();
  void backpropagate();
  int next_batch();
  float test(char* path);
  void train(int total_epochs);
};

void demo(int total_epochs);
int prep_file(char* path, char* out_path);

#endif /* MODULE_H */
