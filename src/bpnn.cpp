//
//  bpnn.cpp
//  Jacobian
//
//  Created by David Freifeld
//

#include "bpnn.hpp"
#include "utils.hpp"
#include <atomic>
#include <chrono>
#include <ctime>
#include <random>

#define SHUFFLED_PATH "./shuffled.txt"
#define VAL_PATH "./test.txt"
#define TRAIN_PATH "./train.txt"

#if (AVX)
#define cwise_product(a,b) avx_product(a, b)
#else
#define cwise_product(a,b) (a).cwiseProduct(b)
#endif

Layer::Layer(int batch_sz, int nodes, float a)
    :alpha(a)
{
    contents = new Eigen::MatrixXf (batch_sz, nodes);
    dZ = new Eigen::MatrixXf (batch_sz, nodes);
    int datalen = batch_sz*nodes;
    for (int i = 0; i < datalen; i++) {
        (*contents)(static_cast<int>(i / nodes),i%nodes) = 0;
        (*dZ)(static_cast<int>(i / nodes),i%nodes) = 0;
    }
    bias = new Eigen::MatrixXf (batch_sz, nodes);
    for (int i = 0; i < nodes; i++) {
        for (int j = 0; j < batch_sz; j++) (*bias)(j, i) = 0;
    }
}

void Layer::operator=(const Layer& that)
{
    activation = that.activation;
    activation_deriv = that.activation_deriv;
    alpha = that.alpha;
    strcpy(activation_str, that.activation_str);
    *contents = *that.contents;
    *v = *that.v;
    *m = *that.m;
    *weights = *that.weights;
    *bias = *that.bias;
    *dZ = *that.dZ;
}

void Layer::init_weights(Layer next)
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

Network::Network(char* path, int batch_sz, float learn_rate, float bias_rate, int regularization, float l, float ratio, bool early_exit, float cutoff)
    :lambda(l), learning_rate(learn_rate), bias_lr(bias_rate), batch_size(batch_sz), reg_type(regularization), early_stop(early_exit), threshold(cutoff)
{
    assert(reg_type == 1 || reg_type == 2); // L1 and L2 are only relevant regularizations
    int total_instances = prep_file(path, SHUFFLED_PATH);
    val_instances = split_file(SHUFFLED_PATH, total_instances, ratio);
    data = fopen(TRAIN_PATH, "r");
    val_data = fopen(VAL_PATH, "r");
    instances = total_instances - val_instances;
    assert(batch_size > 0 || batch_size < instances);
    decay = [this]() -> void {};
    update = [this](std::vector<Eigen::MatrixXf> deltas, int i) {
        *layers[length-2-i].weights -= (learning_rate * deltas[i]);
    };
}

void Network::init_decay(char* type, ...)
{
    va_list args;
    va_start(args, type);
    if (strcmp(type, "step") == 0) {
        float a_0 = va_arg(args, double);
        float k = va_arg(args, double);
        decay = [this, a_0, k]() -> void {
            learning_rate = a_0 * learning_rate/k;
        };
    }
    else if (strcmp(type, "exp") == 0) {
        float a_0 = va_arg(args, double);
        float k = va_arg(args, double);
        decay = [this, a_0, k]() -> void {
            learning_rate = a_0 * exp(-k * epochs);
        };
    }
    else if (strcmp(type, "frac") == 0) {
        float a_0 = va_arg(args, double);
        float k = va_arg(args, double);
        decay = [this, a_0, k]() -> void {
            learning_rate = a_0 / (1+(k * epochs));
        };
    }
    else if (strcmp(type, "linear") == 0) {
        int max_ep = va_arg(args, double);
        decay = [this, max_ep]() -> void {
            learning_rate = 1 - epochs/max_ep;
        };
    }
    else std::cout << "Invalid decay function." << "\n";
    va_end(args);
}

#include "optimizers.cpp"

void Network::add_prelu_layer(int nodes, float a)
{
    length++;
    layers.emplace_back(batch_size, nodes, a);
    strcpy(layers[length-1].activation_str, "prelu");
    layers[length-1].activation = [a](float x) -> float
    {
        if (x > 0) return x;
        else return a * x;
    };
    layers[length-1].activation_deriv = [a](float x) -> float
    {
        if (x > 0) return 1;
        else return a;
    };
}

void Network::add_layer(int nodes, char* name, std::function<float(float)> activation, std::function<float(float)> activation_deriv)
{
    length++;
    layers.emplace_back(batch_size, nodes);
    strcpy(layers[length-1].activation_str, name);
    layers[length-1].activation = activation;
    layers[length-1].activation_deriv = activation_deriv;
}

void Network::initialize()
{
    labels = new Eigen::MatrixXf (batch_size,layers[length-1].contents->cols());
    for (int i = 0; i < length-1; i++) layers[i].init_weights(layers[i+1]);
}

void Network::set_activation(int index, std::function<float(float)> custom, std::function<float(float)> custom_deriv)
{
    layers[index].activation = custom;
    layers[index].activation_deriv = custom_deriv;
}

void Network::feedforward()
{
    for (int i = 0; i < length-1; i++) {
        for (int j = 0; j < layers[i].contents->rows(); j++) {
            if (strcmp(layers[i].activation_str, "linear") == 0) break;
            for (int k = 0; k < layers[i].contents->cols(); k++) {
                (*layers[i].dZ)(j,k) = layers[i].activation_deriv((*layers[i].contents)(j,k));
                (*layers[i].contents)(j,k) = layers[i].activation((*layers[i].contents)(j,k));
            }
        }
        *layers[i+1].contents = (*layers[i].contents) * (*layers[i].weights);
        *layers[i+1].contents += *layers[i+1].bias;
    }
    for (int j = 0; j < layers[length-1].contents->rows(); j++) {
        if (strcmp(layers[length-1].activation_str, "linear") == 0) break;
        for (int k = 0; k < layers[length-1].contents->cols(); k++) {
            (*layers[length-1].dZ)(j,k) = layers[length-1].activation_deriv((*layers[length-1].contents)(j,k));
            (*layers[length-1].contents)(j,k) = layers[length-1].activation((*layers[length-1].contents)(j,k));
        }
    }
    for (int i = 0; i < layers[length-1].contents->rows(); i++) {
        Eigen::MatrixXf m = layers[length-1].contents->block(i,0,1,layers[length-1].contents->cols());
        Eigen::MatrixXf::Index maxRow, maxCol;
        float max = m.maxCoeff(&maxRow, &maxCol);
        m = (m.array() - max).matrix();
#if (AVX)
        float sum = avx_exp(m).sum();
        m = avx_cdiv(avx_exp(m), sum);
#else
        float sum = 0;
        for (int j = 0; j < layers[length-1].contents->cols(); j++) {
            checknan(m(0,j), "input of Softmax operation");
            sum += exp(m(0,j));
        }
        for (int j = 0; j < layers[length-1].contents->cols(); j++) {
            m(0,j) = exp(m(0,j))/sum;
            checknan(m(0,j), "output of Softmax operation");
        }
#endif
        layers[length-1].contents->block(i,0,1,layers[length-1].contents->cols()) = m;
    }
}

void Network::list_net()
{
    std::cout << "-----------------------\nINPUT LAYER (LAYER 0)\n-----------------------\n\n\u001b[31mGENERAL INFO:\x1B[0;37m\nActivation Function: " << layers[0].activation_str << "\n\n\u001b[31mACTIVATIONS:\x1B[0;37m\n" << *layers[0].contents << "\n\n\u001b[31mWEIGHTS:\x1B[0;37m\n" << *layers[0].weights << "\n\n\u001b[31mBIASES:\x1B[0;37m\n" << *layers[0].bias << "\n\n\n";
    for (int i = 1; i < length-1; i++) {
        std::cout << "-----------------------\nLAYER " << i << "\n-----------------------\n\n\u001b[31mGENERAL INFO:\x1B[0;37m\nActivation Function: " << layers[i].activation_str;
        if (strcmp(layers[i].activation_str, "prelu") == 0) std::cout << "\x1B[0;37m\nAlpha (a) value: " << layers[i].alpha;
        std::cout << "\n\n\u001b[31mACTIVATIONS:\x1B[0;37m\n" << *layers[i].contents << "\n\n\u001b[31mBIASES:\x1B[0;37m\n" << *layers[i].bias << "\n\n\u001b[31mWEIGHTS:\x1B[0;37m\n" << *layers[i].weights << "\n\n\n";
    }
    std::cout << "-----------------------\nOUTPUT LAYER (LAYER " << length-1 << ")\n-----------------------\n\n\u001b[31mGENERAL INFO:\x1B[0;37m\nActivation Function: " << layers[length-1].activation_str <<"\n\n\u001b[31mACTIVATIONS:\x1B[0;37m\n" << *layers[length-1].contents << "\n\n\u001b[31BIASES:\x1B[0;37m\n" << *layers[length-1].bias <<  "\n\n\n";
}

float Network::cost()
{
    float sum = 0;
    float reg = 0; // Regularization term
    for (int i = 0; i < layers[length-1].contents->rows(); i++) {
        float tempsum = 0;
        for (int j = 0; j < layers[length-1].contents->cols(); j++) {
            float truth;
            if (j==(*labels)(i,0)) truth = 1;
            else truth = 0;
            if ((*layers[length-1].contents)(i,j) == 0) (*layers[length-1].contents)(i,j) += 0.00001;
            tempsum += truth * log((*layers[length-1].contents)(i,j));
            checknan(tempsum, "summation for row inside cost calculation");
        }
        sum-=tempsum;
        checknan(tempsum, "total summation inside cost calculation");
    }
    for (int i = 0; i < layers.size()-1; i++) {
        if (reg_type == 2) reg += cwise_product(*layers[i].weights,*layers[i].weights).sum();
        else if (reg_type == 1) reg += (layers[i].weights->array().abs().matrix()).sum();
    }
    return ((1.0/batch_size) * sum) + (1/2*lambda*reg);
}

float Network::accuracy()
{
    float correct = 0;
    for (int i = 0; i < layers[length-1].contents->rows(); i++) {
        float ans = -INFINITY;
        float index = -1;
        for (int j = 0; j < layers[length-1].contents->cols(); j++) {
            if ((*layers[length-1].contents)(i, j) > ans) {
                ans = (*layers[length-1].contents)(i, j);
                index = j;
            }
        }
        if ((*labels)(i, 0) == index) correct += 1;
    }
    return (1.0/batch_size) * correct;
}

Eigen::MatrixXf l1_deriv(Eigen::MatrixXf m)
{
    Eigen::MatrixXf r(m.rows(), m.cols());
    for (int i = 0; i < m.rows(); i++) {
        for (int j = 0; j < m.cols(); j++) {
            if (m(i,j) == 0) r(i,j) = 0;
            else r(i,j) = 1;
        }
    }
    return r;
}

void Network::backpropagate()
{
    std::vector<Eigen::MatrixXf> gradients;
    std::vector<Eigen::MatrixXf> deltas;
    Eigen::MatrixXf error (layers[length-1].contents->rows(), layers[length-1].contents->cols());
    for (int i = 0; i < error.rows(); i++) {
        for (int j = 0; j < error.cols(); j++) {
            float truth;
            if (j==(*labels)(i,0)) truth = 1;
            else truth = 0;
            error(i,j) = (*layers[length-1].contents)(i,j) - truth;
            checknan(error(i,j), "gradient of final layer");
        }
    }
    gradients.push_back(error);
    deltas.push_back((*layers[length-2].contents).transpose() * gradients[0]);
    int counter = 1;
    for (int i = length-2; i >= 1; i--) { 
        // TODO: Add nesterov momentum | -p B -t conundrum -t coding -m Without causing segmentation faults.        
        // (*layers[i].weights-((learning_rate * *layers[i].weights) + (0.9 * *layers[i].v))).transpose()
        //grad_calc(gradients, counter, i)
        gradients.push_back(cwise_product(gradients[counter-1] * layers[i].weights->transpose(), *layers[i].dZ));
        deltas.push_back(layers[i-1].contents->transpose() * gradients[counter]);
        counter++;
    }
    for (int i = 0; i < length-1; i++) {
        update(deltas, i);
        if (reg_type == 2) *layers[length-2-i].weights -= ((lambda/batch_size) * (*layers[length-2-i].weights));
        else if (reg_type == 1) *layers[length-2-i].weights -= ((lambda/(2*batch_size)) * l1_deriv(*layers[length-2-i].weights));
        *layers[length-1-i].bias -= bias_lr * gradients[i];
        if (strcmp(layers[length-2-i].activation_str, "prelu") == 0) {
            float sum = 0;
            for (int j = 0; j < layers[length-2-i].contents->rows(); j++) {
                for (int k = 0; k < layers[length-2-i].contents->cols(); k++) {
                    if ((*layers[length-2-i].contents)(j,k)/layers[length-2-i].alpha <= 0) {
                        //  TODO: Review questionable code | -t quality -m Choice of using index i+1 here is sketchy.
                        sum += gradients[i+1](j,k) * (*layers[length-2-i].contents)(j,k)/layers[length-2-i].alpha;
                    }
                }
            }
            layers[length-2-i].alpha += learning_rate * sum;
            float a = layers[length-2-i].alpha;
            layers[length-2-i].activation = [a](float x) -> float
            {
                if (x > 0) return x;
                else return a * x;
            };
            layers[length-2-i].activation_deriv = [a](float x) -> float
            {
                if (x > 0) return 1;
                else return a;
            };
        }
    }
}

void Network::update_layer(float* vals, int datalen, int index)
{
    for (int i = 0; i < datalen; i++) (*layers[index].contents)(static_cast<int>(i / layers[index].contents->cols()), i%layers[index].contents->cols()) = vals[i];
}

#include "data.cpp"

float Network::validate(char* path)
{
    float costsum = 0;
    float accsum = 0;
    for (int i = 0; i <= val_instances-batch_size; i+=batch_size) {
        char line[MAXLINE];
        int inputs = layers[0].contents->cols();
        int datalen = batch_size * inputs;
        float batch[datalen];
        int label = -1;
        for (int i = 0; i < batch_size; i++) {
            fgets(line, MAXLINE, val_data);
            char *p;
            p = strtok(line,",");
            for (int j = 0; j < inputs; j++) {
                batch[j + (i * inputs)] = strtod(p, NULL);
                p = strtok(NULL,",");
            }
            (*labels)(i, 0) = strtod(p, NULL);
        }
        float* batchptr = batch;
        update_layer(batchptr, datalen, 0);
        feedforward();
        costsum += cost();
        accsum += accuracy();
    }
    val_acc = 1.0/(static_cast<float>(val_instances/batch_size)) * accsum;
    val_cost = 1.0/(static_cast<float>(val_instances/batch_size)) * costsum;
    rewind(val_data);
    return 0;
}

void Network::train()
{
    rewind(data);
    float cost_sum = 0;
    float acc_sum = 0;
    for (int i = 0; i <= instances-batch_size; i+=batch_size) {
        if (early_stop == true && get_val_cost() < threshold) return;
        if (i != instances-batch_size) next_batch();
        feedforward();
        backpropagate();
        cost_sum += cost();
        acc_sum += accuracy();
        batches++;
    }
    epoch_acc = 1.0/(static_cast<float>(instances/batch_size)) * acc_sum;
    epoch_cost = 1.0/(static_cast<float>(instances/batch_size)) * cost_sum;
    validate(VAL_PATH);
    if (silenced == false) printf("Epoch %i complete - cost %f - acc %f - val_cost %f - val_acc %f\n", epochs, epoch_cost, epoch_acc, val_cost, val_acc);
    batches=1;
    rewind(data);
    decay();
    epochs++;
}
