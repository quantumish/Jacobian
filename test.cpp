// extern "C" void C_library_function(int x, int y);
#include "/Users/davidfreifeld/Downloads/eigen-3.3.7/Eigen/Dense"
#include <vector>
#include <array>
#include <cstdlib>

struct Matrix
{
public:
  int rows;
  int columns;
  int** contents;

  Matrix(int c, int r, int* vals);
};

Matrix::Matrix(int c, int r, int* vals)
{
  rows = r;
  columns = c;
  int row_count = 0;
  for (int i = 0; i < columns*rows; i++) {
    contents[row_count][i%4] = vals[i];
    if ((i+1)%2 == 0) {
      row_count++;
    }
  }
}

class Node;
class Edge {
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

  Node(int before, int after);
};

Node::Node(int before, int after)
{
  activation = 0;
  bias = rand();
}

class Layer {
  Node* nodes;
  int length;
};

class Network {
  Layer* layers;
  int length;
};

int main()
{
  int bruh[4] = {1,2,3,4};
  Matrix test (2,2, bruh);
}
