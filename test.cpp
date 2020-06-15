// extern "C" void C_library_function(int x, int y);
#include "/Users/davidfreifeld/Downloads/eigen-3.3.7/Eigen/Dense"
#include <vector>
#include <array>
#include <iostream>
#include <string>
#include <cstdio>
#include <fstream>
#include <random>
#include <algorithm>

class Node;
class Edge {
public:
  Node* source;
  Node* end;
  int weight;

  Edge(Node* srcaddr, Node* endaddr);
};

Edge::Edge(Node* srcaddr, Node* endaddr)
{
  source = srcaddr;
  end = endaddr;
  weight = rand();
}

class Node {
public:
  std::vector<Edge> incoming;
  std::vector<Edge> outgoing;
  int activation;
  int bias;

  Node();
};

Node::Node()
{
  activation = 0;
  bias = rand();
}

class Layer {
public:
  Eigen::MatrixXd* contents;
  Eigen::MatrixXd* weights;
  Eigen::MatrixXd* bias;

  Layer(float* vals, int rows, int columns);
  Layer(int rows, int columns);
  void initWeights(Layer next);
};

Layer::Layer(float* vals, int batch_sz, int nodes)
{
  contents = new Eigen::MatrixXd (batch_sz, nodes);
  int datalen = batch_sz*nodes;
  for (int i = 0; i < datalen; i++) {
    (*contents)((int)i / nodes,i%nodes) = vals[i];
  }
  bias = new Eigen::MatrixXd (1, nodes);
  for (int i = 0; i < nodes; i++) {
    (*bias)(0,i) = 0.001;
  }
}

Layer::Layer(int batch_sz, int nodes)
{
  contents = new Eigen::MatrixXd (batch_sz, nodes);
  int datalen = batch_sz*nodes;
  for (int i = 0; i < datalen; i++) {
    (*contents)((int)i / nodes,i%nodes) = 0;
  }
  bias = new Eigen::MatrixXd (1, nodes);
  for (int i = 0; i < nodes; i++) {
    (*bias)(0,i) = 0.001;
  }
}

void Layer::initWeights(Layer next)
{
  weights = new Eigen::MatrixXd (contents->cols(), next.contents->cols());
  int nodes = weights->cols();
  for (int i = 0; i < (weights->rows()*weights->cols()); i++) {
    (*weights)((int)i / nodes, i%nodes) = rand() / double(RAND_MAX);
  }
}

class Network {
public:
  char* fpath;

  std::vector<Layer> layers;
  int length;

  float learning_rate;
  int batch_size;
  int batches;
  Eigen::MatrixXd* labels;

  Network(char* path, int inputs, int hidden, int outputs, int neurons, int batch_sz, float rate);
  void update_layer(float* vals, int datalen, int index);

  Eigen::MatrixXd activate(Eigen::MatrixXd matrix);
  void feedforward();
  void list_net();

  float cost();
  float gradient(int mode, int layer, int node);
  void backpropagate();
  int next_batch();
  void test(char* path);
};

Network::Network(char* path, int inputs, int hidden, int outputs, int neurons, int batch_sz, float rate)
{
  learning_rate = rate;
  fpath = path;
  length = hidden + 2;
  batch_size = batch_sz;
  FILE* fptr = fopen(path, "r");
  int datalen = batch_sz*inputs;
  float batch[datalen];
  labels = new Eigen::MatrixXd (batch_sz, 1);
  int label;
  char line[1024] = {' '};
  for (int i = 0; i < batch_sz; i++) {
    fgets(line, 1024, fptr);
    sscanf(line, "%f,%f,%f,%f,%i", &batch[0+(i*inputs)], &batch[1+(i*inputs)], &batch[2+(i*inputs)], &batch[3+(i*inputs)], &label);
    (*labels)(i,0) = label;
  }
  float* batchptr = batch;
  layers.emplace_back(batchptr, batch_sz, inputs);
  for (int i = 0; i < hidden; i++) {
    layers.emplace_back(batch_sz, neurons);
  }
  layers.emplace_back(batch_sz, outputs);
  for (int i = 0; i < hidden+1; i++) {
    layers[i].initWeights(layers[i+1]);
  }
}

Eigen::MatrixXd Network::activate(Eigen::MatrixXd matrix)
{
  int nodes = matrix.cols();
  for (int i = 0; i < (matrix.rows()*matrix.cols()); i++) {
    (matrix)((float)i / nodes, i%nodes) = 1.0/(1+exp(-(matrix)((float)i / nodes, i%nodes)));
  }
  return matrix;
}

void Network::feedforward()
{
  for (int i = 0; i < length-1; i++) {
    *layers[i+1].contents = (*layers[i].contents) * (*layers[i].weights);
    for (int j = 0; j < layers[i+1].contents->rows(); j++) {
      // layers[i+1].contents->row(j) += *layers[i+1].bias;
    }
    *layers[i+1].contents = activate(*layers[i+1].contents);
  }
}

void Network::list_net()
{
  for (int i = 0; i < length-1; i++) {
    std::cout << " LAYER " << i << "\n\n" << *layers[i].contents << "\n\n AND BIAS\n" << *layers[i].bias << "\n\n W/ WEIGHTS \n" << *layers[i].weights << "\n\n\n";
  }
  std::cout << " LAYER " << length-1 << "\n\n" << *layers[length-1].contents << "\n\n AND BIAS\n" << *layers[length-1].bias <<  "\n\n\n";
}

float Network::cost()
{
  float sum = 0;
  for (int i = 0; i < layers[length-1].contents->rows(); i++) {
    sum += pow((*labels)(i, 0)*100 - (*layers[length-1].contents)(i, 0)*100,2);
  }
  return (1.0/batch_size) * sum;
}

void Network::backpropagate()
{
  int N = batch_size;

  std::vector<Eigen::MatrixXd> gradients;
  std::vector<Eigen::MatrixXd> errors;
  Eigen::MatrixXd e = ((*layers[length-1].contents ) - (*labels)) * ((*layers[length-1].contents ) - (*labels));
  Eigen::MatrixXd D (layers[length-1].contents->cols(), layers[length-1].contents->cols());
  for (int i = 0; i < layers[length-1].contents->cols(); i++) {
    for (int j = 0; j < layers[length - 1].contents->cols(); j++) {
      D(j, i) = 0;
    }
  }
  for (int i = 0; i < layers[length-1].contents->cols(); i++) {
    D(i, i) = (*layers[length-1].contents)(0, i) * (1 - (*layers[length-1].contents)(0, i));
  }
  gradients.push_back(layers[length-2].contents->transpose() * (D * e));
  // std::cout << gradients[0] << "\n\n";
  // std::cout << D << "\n\nTHEN\n\n" << layers[length-2].contents->transpose() << "\n\nNEXT\n\n" << e << "\n\nSO\n\n" << gradients[0] << "\n\n\n\n\n";
  int counter = 0;
  for (int i = length-2; i >= 1; i--) {
    Eigen::MatrixXd D_l (layers[i].contents->cols(), layers[i].contents->cols());
    // std::cout << i << " Aye!\n";
    for (int j = 0; j < layers[i].contents->cols(); j++) {
      for (int k = 0; k < layers[i].contents->cols(); k++) {
        D_l(k, j) = 0;
      }
    }
    for (int j = 0; j < layers[i].contents->cols(); j++) {
      D_l(j, j) = (*layers[i].contents)(0,j) * (1 - (*layers[i].contents)(0,j));
    }
    std::cout << D_l << "\n\nTHEN\n\n" << layers[i].weights->transpose() << "\n\nNEXT\n\n" << gradients[counter] << "\n\n";

    Eigen::MatrixXd e_l = D_l * ( gradients[counter] * layers[i].weights->transpose());
    // std::cout << "\n\nSO\n\n" << e_l <<  "\n\n\n\n\n";
    gradients.push_back(e_l);
    counter++;
  }
  for (int i = 1; i < gradients.size(); i++) {
    Eigen::MatrixXd gradient = gradients[i];
    // printf("%i\n", length-1-i);
    std::cout << *layers[length-2-i].weights << " \n\n and \n\n " << gradients[i] << "\n\n";
    // std::cout <<"YAY?\n";
    *layers[length-2-i].weights -= (gradients[i]);
  }
}

void Network::update_layer(float* vals, int datalen, int index)
{
  for (int i = 0; i < datalen; i++) {
    (*layers[index].contents)((int)i / layers[index].contents->cols(),i%layers[index].contents->cols()) = vals[i];
  }
}

int Network::next_batch()
{
  FILE* fptr = fopen(fpath, "r");
  char line[1024] = {' '};
  int inputs = layers[0].contents->cols();
  int datalen = batch_size * inputs;
  float batch[datalen];
  int label = 100;
  for (int i = 0; i < batch_size*batches + 1; i++) {
    if (fgets(line, 1024, fptr)==NULL) {
      break;
    }
    if (i >= batches) {
      for (int j = 0; j < batch_size; j++) {
        fgets(line, 1024, fptr);
        sscanf(line, "%f,%f,%f,%f,%i", &batch[0 + (j * inputs)],
               &batch[1 + (j * inputs)], &batch[2 + (j * inputs)],
               &batch[3 + (j * inputs)], &label);
        (*labels)(j, 0) = label;
      }
    }
  }
  float* batchptr = batch;
  update_layer(batchptr, datalen, 0);
  fclose(fptr);
  return 0;
}

int prep_file(char* path)
{
  FILE* rptr = fopen(path, "r");
  char line[1024];
  std::vector<std::string> lines;
  int count = 0;
  while (fgets(line, 1024, rptr) != NULL) {
    lines.emplace_back(line);
    count++;
  }
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(lines.begin(), lines.end(), g);
  fclose(rptr);
  std::ofstream out("./shuffled.txt");
  for (int i = 0; i < lines.size(); i++) {
    out << lines[i];
  }
  return count;
}

void Network::test(char* path)
{
  int rounds = 1;
  int exit = 0;
  float totalcost = -1;
  int linecount = prep_file(path);
  while (exit == 0) {
    FILE* fptr = fopen(fpath, "r");
    char line[1024] = {' '};
    int inputs = layers[0].contents->cols();
    int datalen = batch_size * inputs;
    float batch[datalen];
    for (int i = 0; i < batch_size*rounds + 1; i++) {
      if (fgets(line, 1024, fptr)==NULL) {
        exit = -1;
      }
      if (i >= rounds) {
        for (int j = 0; j < batch_size; j++) {
          fgets(line, 1024, fptr);
          sscanf(line, "%f,%f,%f,%f,*i", &batch[0 + (j * inputs)], &batch[1 + (j * inputs)], &batch[2 + (j * inputs)], &batch[3 + (j * inputs)]);
        }
      }
    }
    float *batchptr = batch;
    update_layer(batchptr, datalen, 0);
    fclose(fptr);
    feedforward();
    totalcost += cost();
    rounds++;
  }
  std::cout << "TEST COST: " << 1.0/((float) linecount) * totalcost << "\n";
}

void demo()
{
  // std::cout << "\n\n\n";
  int linecount = prep_file("./data_banknote_authentication.txt");
  Network net ("./shuffled.txt", 4, 2, 1, 5, 1, 1);
  float epoch_cost = 1000;
  int epochs = 0;
  net.batches= 1;
  while (epochs < 1) {
    // int linecount = prep_file("./data_banknote_authentication.txt");
    float cost_sum = 0;
    for (int i = 0; i < linecount; i++) {
      net.feedforward();
      net.backpropagate();
      cost_sum += net.cost();
      // std::cout << net.cost() << " as it is " << net.labels[0] << " vs " << *net.layers[net.length-1].contents << "\n";
      // net.list_net();
      net.batches++;
      int exit = net.next_batch();
      if (exit == -1) {
        break;
      }
    }
    net.batches=1;
    epoch_cost = 1.0/((float) linecount) * cost_sum;
    printf("EPOCH %i: Cost is %f for %i instances.\n", epochs, epoch_cost, linecount);
    std::cout << *net.layers[net.length-2].weights << "\n\n";
    epochs++;
  }
  net.test("./test.txt");
  net.feedforward();
}

int main()
{
  demo();
}
