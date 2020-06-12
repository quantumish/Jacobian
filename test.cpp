// extern "C" void C_library_function(int x, int y);
#include "/Users/davidfreifeld/Downloads/eigen-3.3.7/Eigen/Dense"
#include <vector>
#include <array>
#include <cstdlib>

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

  Layer(int* vals, int rows, int columns);
  void initWeights(Layer next, int batch_sz);
};

Layer::Layer(int* vals, int batch_sz, int nodes)
{
  contents = new Eigen::MatrixXd (batch_sz, nodes);
  for (int i = 0; i < batch_sz*nodes; i++) {
    *contents << vals[i];
  }
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
  int batch[batch_sz * 4];
  for (int i = 0; i < batch_sz; i++) {
    fscanf(fptr, "%d,%d,%d,%d,*d", &batch[0+i], &batch[1+i], &batch[2+i], &batch[3+i]);
  }
  layers.emplace_back(batch, batch_sz, inputs);
  // for (int i = 0; i < hidden; i++) {
  //   layers.emplace_back(neurons);
  // }
  // layers.emplace_back(outputs);
}


int main()
{
  Network net ("../mapreduce/testing.txt", 4, 2, 2, 5, 10);
}
