
#include "utils.hpp"
#include "cnn.hpp"
#include "bpnn.hpp"

class Capsule
{
    Capsule* children;
    Capsule* parents;
}

class CapsNet
{
public:
    std::vector<ConvLayer> conv_layers;
    std::vector<Capsule> primary_caps;
    std::vector<Capsule> class_caps;
    CapsNet(char* path);
    void add_conv_layer();
    void add_capsule();
    void feedforward();
    void routing();
};
