#include <functional>
#include <vector>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cmath>

#include "utils.hpp"

namespace Jacobian {

struct Node
{
public:
    Eigen::MatrixXf val;
    std::vector<Node*> args;
    std::function<void(Node&)> op;
    bool calced;
    Node(float v);
    Node(Eigen::MatrixXf v);
    Node(std::function<void(Node&)> f, std::vector<Node*> a);
};

Node::Node(float v) : op(nullptr), val(1,1), args{}, calced(true)
{
    val(0,0) = v;
}

Node::Node(Eigen::MatrixXf v) : op(nullptr), val(v), args{}, calced(true) {}
Node::Node(std::function<void(Node&)> f, std::vector<Node*> a)
    :op(f), val(Eigen::MatrixXf::Zero(1,1)), args(a), calced(false) {}



namespace internal {
void multiply(Node& node)
{
    node.val = node.args[0]->val;
    for (size_t i = 1; i < node.args.size(); i++) {
        node.val*node.args[i]->val;
    }
}

void add(Node& node)
{
    for (size_t i = 0; i < node.args.size(); i++) {
        node.val*node.args[i]->val;
    }
}

void tanh(Node& node)
{
    for (int i = 0; i < node.val.rows(); i++) {
        for (int j = 0; j < node.val.cols(); j++) {
            node.val(i,j) = ftanh(node.val(i,j));
        }
    }
}
}


Node* multiply(Node* a, Node* b) {return new Node(internal::multiply, {a,b});}
Node* add(Node* a, Node* b) {return new Node(internal::add, {a,b});}
Node* tanh(Node* a) {return new Node(internal::tanh, {a});}


class Graph
{
public:
    Graph();
    Node* define(float a);
    Node* define(Eigen::MatrixXf a);
    void eval(Node* back);
};

Node* Graph::define(float a) { return new Node(a); }
Node* Graph::define(Eigen::MatrixXf a) { return new Node(a); }


Graph::Graph() {}

void Graph::eval(Node* node)
{
    for (Node* arg : node->args) {
        if (arg->calced == false) {
            eval(arg);
        }
    }
    node->op(*node);
}
}

int main()
{
    Jacobian::Graph g {};

    // Define input biases
    Eigen::MatrixXf _b = Eigen::MatrixXf::Constant(5,4,0);
    auto b = g.define(_b);

    // Define input layer
    Eigen::MatrixXf _x(5,3);
    _x << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15;
    auto x = g.define(_x);

    // Define weights
    Eigen::MatrixXf _W = Eigen::MatrixXf::Constant(3,4,1);
    auto W = g.define(_W);

    // Define hidden layer output
    auto h = add(b, multiply(x, W));

    // Define final weights
    Eigen::MatrixXf _V = Eigen::MatrixXf::Constant(4,2,1);
    auto V = g.define(_V);

    // Define final biases
    Eigen::MatrixXf _a = Eigen::MatrixXf::Constant(5,2,0);
    auto a = g.define(_a);

    // Define output layer
    auto y = add(a, multiply(h, V));

    // Feedforward.
    g.eval(y);

    // Get output.
    std::cout << y->val << '\n';
}
