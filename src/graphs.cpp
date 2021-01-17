#include <functional>
#include <vector>

enum class Operation {add, multiply};
    
class Node
{
    Operation op;
    float value;
    std::vector<Node*> children;
};

Node* add(Node* a, Node* b)
{
    Node* result = new Node;
    a->children.push_back(result);
    b->children.push_back(result);
    result->op = Operation::add;
    return result;
}

Node* multiply(Node* a, Node* b)
{
    Node* result = new Node;
    a->children.push_back(result);
    b->children.push_back(result);
    result->op = Operation::multiply;
    return result;
};

Node* define(float a)
{
    Node* result = new Node;
    result->value = a;
}

int main()
{
    auto a = define(5);
    auto b = define(3.2);
    auto c = define(2);
    auto d = add(a, multiply(b, c));
    
}
