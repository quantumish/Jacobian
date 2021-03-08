#include <functional>
#include <vector>

enum class Operation {NONE, add, multiply};
    
struct Node
{    
    Operation op;
    float value;
    std::vector<Node*> args;
    Node(float v);
    Node(Operation o, std::vector<Node*> a);
};

Node::Node(float v) : op(Operation::NONE), value(v), args{} {}
Node::Node(Operation o, std::vector<Node*> a) : op(o), value(), args(a) {}

Node* add(Node* a, Node* b)
{
    Node* result = new Node(Operation::add, {a,b});
    result->args.push_back(a);
    result->args.push_back(b);
    result->op = Operation::add;
    return result;
    
}

Node* multiply(Node* a, Node* b)
{
    Node* result = new Node(Operation::multiply, {a,b});
    result->args.push_back(a);
    result->args.push_back(b);
    result->op = Operation::multiply;
    return result;
}

class Graph
{
    std::vector<Node> input;
public:
    Graph();
    Node* define(float a);
    void start(Node* back);
};

Node* Graph::define(float a)
{
    input.emplace_back(a);
    return &input[input.size()-1];
}



void Graph::start(Node* back)
{
    Node* cur = back;
    for (size_t i = 0; i < back->args.size(); i++) {
	
    }
}

int main()
{
    Graph g {};
    auto a = g.define(5);
    auto b = g.define(3.2);
    auto c = g.define(2);
    auto d = add(a, multiply(b, c));
    g.start(d);
}
