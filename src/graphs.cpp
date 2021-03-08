#include <functional>
#include <vector>
#include <iostream>
#include <eigen3/Eigen/Dense>

enum class Operation {NONE, add, multiply};

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
    std::cout << back->args.size() << "\n";
    for (Node* arg : back->args) {
	std::cout << "At " << back << ", checking if arg " << arg << " is evaluated..."; 
	if (arg->calced == false) {
	    std::cout << "nope!" << "\n";
	    eval(arg);	    
	}
	std::cout << "yup!" << "\n";	
    }
    std::cout << "Calcing...";
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
	    back->val*=back->args[i]->val(0,0);
	}
	break;
    }
    std::cout << back->val << "\n";	
}

int main()
{
    Graph g {};
    Eigen::MatrixXf n = Eigen::MatrixXf::Constant(3,3,1);
    auto a = g.define(n);
    Eigen::MatrixXf m(3,3);
    m << 1,2,3,4,5,6,7,8,9;
    auto b = g.define(m);
    auto c = g.define(2);
    auto d = add(a, multiply(b, c));
    std::cout << "A: " << a << "\n";
    std::cout << "B: " << b << "\n";
    std::cout << "C: " << c << "\n";
    std::cout << "D: " << d << "\n\n";
    g.eval(d);
    std::cout << d->val << '\n';
}
