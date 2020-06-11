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
  std::vector<Node> nodes;
  int length;

  Layer(int len);
};

Layer::Layer(int len)
{
  length = len;
}

class Network {
public:
  std::vector<Layer> layers;
  int length;

  Network(char* path, int inputs, int hidden, int outputs, int neurons);
};

Network::Network(char* path, int inputs, int hidden, int outputs, int neurons)
{
  fscanf("%s,%s,%s,%s,*s");
  layers.emplace_back(inputs);
  layers[0].nodes.emplace_back();
  for (int i = 0; i < hidden; i++) {
    layers.emplace_back(neurons);
    layers[i+1].nodes.emplace_back();
  }
  layers.emplace_back(outputs);
  layers[layers.size()-1].nodes.emplace_back();
}


int main()
{
  Network net ("abc", 4, 2, 2, 5);
}
