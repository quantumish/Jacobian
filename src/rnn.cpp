#include "bpnn.hpp"

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
    for (int i = 0; i < (rec_weights->cols()*rec_weights->cols()); i++) {
        std::random_device rd;
        std::mt19937 gen(rd());
        (*rec_weights)(static_cast<int>(i / columns), i%columns) = d(gen);
    }
    for (int i = 0; i < rows*columns; i++) {
        (*s)(static_cast<int>(i / nodes),i%nodes) = 0;
    }
}

void init_weights(RecurrentLayer next)
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
        (*weights)(static_cast<int>(i / nodes), i%nodes) = d(gen);
        (*v)(static_cast<int>(i / nodes), i%nodes) = 0;
        (*m)(static_cast<int>(i / nodes), i%nodes) = 0;
    }
}
    
class RNN : public Network {
public:
    std::vector<RecurrentLayer> layers;
    void feedforward();
    void backpropagate();
    RNN(char* path, int batch_sz, float learn_rate, float bias_rate, Regularization regularization, float l, float ratio, bool early_exit=true, float cutoff=0);

};

RNN::RNN(char* path, int batch_sz, float learn_rate, float bias_rate, Regularization regularization, float l, float ratio, bool early_exit=true, float cutoff=0)
    :Network(path, batch_sz, learn_rate, bias_rate, regularization, l, ratio, early_exit, cutoff)
{}

void RNN::feedforward()
{
    for (int i = 0; i < length-1; i++) {
        for (int j = 0; j < layers[i].contents->rows(); j++) {
            if (strcmp(layers[i].activation_str, "linear") == 0) break;
            for (int k = 0; k < layers[i].contents->cols(); k++) {
                (*layers[i].dZ)(j,k) = layers[i].activation_deriv((*layers[i].contents)(j,k));
                (*layers[i].contents)(j,k) = layers[i].activation((*layers[i].contents)(j,k));
            }
        }
        *layers[i+1].contents = ((*layers[i].s) * (*layers[i].rec_weights)) + ((*layers[i].contents) * (*layers[i].weights));
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
