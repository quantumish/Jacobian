#include <functional>
#include <vector>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cmath>

enum class Operation {NONE, add, multiply, tanh};

struct Node
{    
    Operation op;
    Eigen::MatrixXf val;
    std::vector<Node*> args;
    bool calced;
    Node(float v);
    Node(Eigen::MatrixXf v);
    Node(Operation o, std::vector<Node*> a);
};

Node::Node(float v) : op(Operation::NONE), val(1,1), args{}, calced(true)
{
    val(0,0) = v;
}

namespace Jacobian {
    Node* tanh(Node* a) {return new Node(Operation::tanh, {a});}
}

Node::Node(Eigen::MatrixXf v) : op(Operation::NONE), val(v), args{}, calced(true) {}
Node::Node(Operation o, std::vector<Node*> a) : op(o), val(Eigen::MatrixXf::Zero(1,1)), args(a), calced(false) {}

Node* add(Node* a, Node* b) {return new Node(Operation::add, {a,b});}
Node* multiply(Node* a, Node* b) {return new Node(Operation::multiply, {a,b});}

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

void Graph::eval(Node* back)
{
    for (Node* arg : back->args) {
	if (arg->calced == false) {
	    eval(arg);	    
	}
    }
    // std::cout << back->val << "\n\n";
    switch (back->op) {
    case Operation::NONE:
	break;
    case Operation::add:
	back->val = back->args[0]->val;
	for (size_t i = 1; i < back->args.size(); i++) {
	    back->val+=back->args[i]->val;
	}
	break;
    case Operation::multiply:	
	back->val = back->args[0]->val;
	for (size_t i = 1; i < back->args.size(); i++) {
	    back->val*=back->args[i]->val;
	}
	break;
    case Operation::tanh:
	for (int i = 0; i < back->val.rows(); i++) {
            for (int j = 0; j < back->val.cols(); j++) {
                back->val(i,j) = tanh(back->val(i,j));
            }
        }
	back->val = back->args[0]->val;	
	break;

    }
}

int main()
{    
    Graph g {};

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
