#include "bpnn.hpp"

class RecurrentLayer : public Layer {
public:
    Eigen::MatrixXf* rec_weights;
    void init_weights(RecurrentLayer next);
    RecurrentLayer(int rows, int columns, float a=0);
};

RecurrentLayer::RecurrentLayer(int rows, int columns, float a)
    :Layer(rows, columns, a)
{
    rec_weights = new Eigen::MatrixXf(contents->cols(), contents->cols());
    for (int i = 0; i < (rec_weights->rows()*rec_weights->cols()); i++) {
        std::random_device rd;
        std::mt19937 gen(rd()); 
        (*rec_weights)(static_cast<int>(i / columns), i%columns) = d(gen);
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


