#include <functional>
#include <vector>
#include <iostream>

enum class Operation {NONE, add, multiply};
    
struct Node
{    
    Operation op;
    float val;
    std::vector<Node*> args;
    bool calced;
    Node(float v);
    Node(Operation o, std::vector<Node*> a);
};

Node::Node(float v) : op(Operation::NONE), val(v), args{}, calced(true) {}
Node::Node(Operation o, std::vector<Node*> a) : op(o), val(0), args(a), calced(false) {}

Node* add(Node* a, Node* b)
{
    Node* result = new Node(Operation::add, {a,b});
    return result;
    
}

Node* multiply(Node* a, Node* b)
{
    Node* result = new Node(Operation::multiply, {a,b});
    return result;
}

class Graph
{
public:
    Graph();
    Node* define(float a);
    void eval(Node* back);
};

Node* Graph::define(float a)
{
    Node* n = new Node(a);
    return n;
}

Graph::Graph()
{
}

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
	for (Node* arg : back->args) back->val+=arg->val;
	break;
    case Operation::multiply:
	back->val = back->args[0]->val;
	for (size_t i = 1; i < back->args.size(); i++) {
	    back->val*=back->args[i]->val;
	}
	break;	
    }
    std::cout << back->val << "\n";	
}

int main()
{
    Graph g {};
    auto a = g.define(5);
    auto b = g.define(3.2);
    auto c = g.define(2);
    auto d = add(a, multiply(b, c));
    std::cout << "A: " << a << "\n";
    std::cout << "B: " << b << "\n";
    std::cout << "C: " << c << "\n";
    std::cout << "D: " << d << "\n\n";
    g.eval(d);
    std::cout << d->val << '\n';
}
