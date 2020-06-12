// extern "C" void C_library_function(int x, int y);
#include "/Users/davidfreifeld/Downloads/eigen-3.3.7/Eigen/Dense"
#include <vector>
#include <array>
#include <iostream>
#include <cstdlib>
#include <cstdio>

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

  Layer(float* vals, int rows, int columns);
  void initWeights(Layer next, int batch_sz);
};

Layer::Layer(float* vals, int batch_sz, int nodes)
{
  contents = new Eigen::MatrixXd (batch_sz, nodes);
  int datalen = batch_sz*nodes;
  for (int i = 0; i < datalen; i++) {
    (*contents)((int)i / 4,i%nodes) = vals[i];
  }
  std::cout << *contents << "\n";
}

void Layer::initWeights(Layer next, int batch_sz)
{
  weights = new Eigen::MatrixXd (contents->rows(), next.contents->cols());
  for (int i = 0; i < (weights->rows()*weights->cols()); i++) {
    *weights << rand();
  }
}

class Network {
public:
  std::vector<Layer> layers;
  int length;

  Network(char* path, int inputs, int hidden, int outputs, int neurons, int batch_sz);
};

Network::Network(char* path, int inputs, int hidden, int outputs, int neurons, int batch_sz)
{
  FILE* fptr = fopen(path, "r");
  int datalen = batch_sz*inputs;
  float batch[datalen];
  char line[1024] = {' '};
  for (int i = 0; i < batch_sz; i++) {
    fgets(line, 1024, fptr);
    sscanf(line, "%f,%f,%f,%f,*f", &batch[0+(i*inputs)], &batch[1+(i*inputs)], &batch[2+(i*inputs)], &batch[3+(i*inputs)]);
  }
  float* batchptr = batch;
  // for (int i = 0; i < datalen; i++) {
  //   printf("%f (vs %f) at %x\n", batchptr[i], batch[i], batchptr);
  // }
  layers.emplace_back(batchptr, batch_sz, inputs);
  // for (int i = 0; i < hidden; i++) {
  //   layers.emplace_back(neurons);
  // }
  // layers.emplace_back(outputs);
}


int main()
{
  Network net ("./data_banknote_authentication.txt", 4, 2, 2, 5, 10);
}
