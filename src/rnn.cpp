#include "bpnn.hpp"
#include "utils.hpp"

class RecurrentLayer : public Layer {
public:
    Eigen::MatrixXf* s;
    Eigen::MatrixXf* rec_weights;
    void init_weights(RecurrentLayer next);
    RecurrentLayer(int rows, int columns, float a=0);
};

RecurrentLayer::RecurrentLayer(int rows, int columns, float a)
    :Layer(rows, columns, a)
{
    s = new Eigen::MatrixXf(contents->rows(), contents->cols());
    rec_weights = new Eigen::MatrixXf(contents->cols(), contents->cols());
    int n = contents->cols() * 2;
    std::normal_distribution<float> d(0,sqrt(1.0/n));
    for (int i = 0; i < (rec_weights->cols()*rec_weights->cols()); i++) {
        std::random_device rd;
        std::mt19937 gen(rd());
        (*rec_weights)(static_cast<int>(i / columns), i%columns) = d(gen);
    }
    for (int i = 0; i < rows*columns; i++) {
        (*s)(static_cast<int>(i / columns),i%columns) = 0;
    }
}

void RecurrentLayer::init_weights(RecurrentLayer next)
{
    v = new Eigen::MatrixXf (contents->cols(), next.contents->cols());
    m = new Eigen::MatrixXf (contents->cols(), next.contents->cols());
    weights = new Eigen::MatrixXf (contents->cols(), next.contents->cols());
    int nodes = weights->cols();
    int n = contents->cols() + next.contents->cols();
    std::normal_distribution<float> d(0,sqrt(1.0/n));
    for (int i = 0; i < (weights->rows()*weights->cols()); i++) {
        std::random_device rd;
        std::mt19937 gen(rd()); 
        (*weights)(static_cast<int>(i / weights->cols()), i%weights->cols()) = d(gen);
        (*v)(static_cast<int>(i / weights->cols()), i%weights->cols()) = 0;
        (*m)(static_cast<int>(i / weights->cols()), i%weights->cols()) = 0;
    }
}
    
class RNN : public Network {
public:
    std::vector<RecurrentLayer> layers;
    int length = 0;
    void add_layer(int nodes, char* name, std::function<float(float)> activation, std::function<float(float)> activation_deriv);
    void initialize();
    void feedforward();
    void backpropagate();
    RNN(char* path, int batch_sz, float learn_rate, float bias_rate, Regularization regularization, float l, float ratio, bool early_exit=true, float cutoff=0);

};

RNN::RNN(char* path, int batch_sz, float learn_rate, float bias_rate, Regularization regularization, float l, float ratio, bool early_exit, float cutoff)
    :Network(path, batch_sz, learn_rate, bias_rate, regularization, l, ratio, early_exit, cutoff)
{}

void RNN::add_layer(int nodes, char* name, std::function<float(float)> activation, std::function<float(float)> activation_deriv)
{
    length++;
    layers.emplace_back(batch_size, nodes);
    strcpy(layers[length-1].activation_str, name);
    layers[length-1].activation = activation;
    layers[length-1].activation_deriv = activation_deriv;
}

void RNN::initialize()
{
    labels = new Eigen::MatrixXf (batch_size,layers[length-1].contents->cols());
    for (int i = 0; i < length-1; i++) layers[i].init_weights(layers[i+1]);
}

void RNN::feedforward()
{
    for (int i = 0; i < length-1; i++) {
        *layers[i].contents = ((*layers[i].s) * (*layers[i].rec_weights));
        for (int j = 0; j < layers[i].contents->rows(); j++) {
            if (strcmp(layers[i].activation_str, "linear") == 0) break;
            for (int k = 0; k < layers[i].contents->cols(); k++) {
                (*layers[i].dZ)(j,k) = layers[i].activation_deriv((*layers[i].contents)(j,k));
                (*layers[i].contents)(j,k) = layers[i].activation((*layers[i].contents)(j,k));
            }
        }
        *layers[i+1].contents =  ((*layers[i].contents) * (*layers[i].weights));
        *layers[i+1].contents += *layers[i+1].bias;
    }
    for (int j = 0; j < layers[length-1].contents->rows(); j++) {
        if (strcmp(layers[length-1].activation_str, "linear") == 0) break;
        for (int k = 0; k < layers[length-1].contents->cols(); k++) {
            (*layers[length-1].dZ)(j,k) = layers[length-1].activation_deriv((*layers[length-1].contents)(j,k));
            (*layers[length-1].contents)(j,k) = layers[length-1].activation((*layers[length-1].contents)(j,k));
        }
    }
}

int main()
{
    RNN rnn ("../data_banknote_authentication.txt", 10, 0.0155, 0.03, L2, 0, 0.9);
    rnn.add_layer(4, "linear", linear, linear_deriv);
    rnn.add_layer(5, "lecun_tanh", lecun_tanh, lecun_tanh_deriv);
    rnn.add_layer(2, "linear", linear, linear_deriv);
    rnn.initialize();
    rnn.feedforward();
}
