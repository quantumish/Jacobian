//
//  bpnn.cpp
//  Jacobian
//
//  Created by David Freifeld
//

#include "bpnn.hpp"
#include "utils.hpp"
#include <ctime>
#include <random>
#include <Eigen/MatrixFunctions>
#include <Eigen/unsupported/CXX11/Tensor>
#include <indicators/progress_bar.hpp>

#define SHUFFLED_PATH "./shuffled.txt"
#define VAL_PATH "./test.txt"
#define TRAIN_PATH "./train.txt"

//#include "checks.cpp"

Layer::Layer(int batch_sz, int nodes, float a)
    :alpha(a)
{
    contents = new Eigen::MatrixXf (batch_sz, nodes);
    dZ = new Eigen::MatrixXf (batch_sz, nodes);
    int datalen = batch_sz*nodes;
    for (int i = 0; i < datalen; i++) {
        (*contents)((int)i / nodes,i%nodes) = 0;
        (*dZ)((int)i / nodes,i%nodes) = 0;
    }
    bias = new Eigen::MatrixXf (batch_sz, nodes);
    for (int i = 0; i < nodes; i++) {
        for (int j = 0; j < batch_sz; j++) {
            (*bias)(j, i) = 0;
        }
    }
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
        (*weights)((int)i / nodes, i%nodes) = d(gen);
        (*v)((int)i / nodes, i%nodes) = 0;
        (*m)((int)i / nodes, i%nodes) = 0;
    }
}

Network::Network(char* path, int batch_sz, float learn_rate, float bias_rate, int regularization, float l, float ratio, bool early_exit, float cutoff)
    :lambda(l), learning_rate(learn_rate), bias_lr(bias_rate), batch_size(batch_sz), reg_type(regularization), early_stop(early_exit), threshold(cutoff)
{
    assert(reg_type == 1 || reg_type == 2); // L1 and L2 are only relevant regularizations
    int total_instances = prep_file(path, SHUFFLED_PATH);
    val_instances = split_file(SHUFFLED_PATH, total_instances, ratio);
    instances = total_instances - val_instances;
    data = fopen(TRAIN_PATH, "r");
    val_data = fopen(VAL_PATH, "r");
    decay = [this]() -> void {
        learning_rate = learning_rate;
    };
    grad_calc = [this](std::vector<Eigen::MatrixXf> gradients, int i, int counter) -> void {
        gradients.push_back(avx_product(gradients[counter-1] * layers[i].weights->transpose(), *layers[i].dZ));
    };
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
    if (strcmp(type, "exp") == 0) {
        float a_0 = va_arg(args, double);
        float k = va_arg(args, double);
        decay = [this, a_0, k]() -> void {
            learning_rate = a_0 * exp(-k * epochs);
        };
    }
    if (strcmp(type, "frac") == 0) {
        float a_0 = va_arg(args, double);
        float k = va_arg(args, double);
        decay = [this, a_0, k]() -> void {
            learning_rate = a_0 / (1+(k * epochs));
        };
    }
    if (strcmp(type, "linear") == 0) {
        int max_ep = va_arg(args, double);
        decay = [this, max_ep]() -> void {
            learning_rate = 1 - epochs/max_ep;
        };
    }
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

// Gross code inbound
void Network::add_layer(int nodes, char* name)
{
    length++;
    layers.emplace_back(batch_size, nodes);
    strcpy(layers[length-1].activation_str, name);
    if (strcmp(name, "sigmoid") == 0) {
        layers[length-1].activation = sigmoid;
        layers[length-1].activation_deriv = sigmoid_deriv;
    }
    else if (strcmp(name, "linear") == 0) {
        layers[length-1].activation = linear;
        layers[length-1].activation_deriv = linear_deriv;
    }
    else if (strcmp(name, "step") == 0) {
        layers[length-1].activation = step;
        layers[length-1].activation_deriv = step_deriv;
    }
    else if (strcmp(name, "lecun_tanh") == 0) {
        layers[length-1].activation = lecun_tanh;
        layers[length-1].activation_deriv = lecun_tanh_deriv;
    }
    else if (strcmp(name, "inverse_logit") == 0) {
        layers[length-1].activation = inverse_logit;
        layers[length-1].activation_deriv = inverse_logit_deriv;
    }
    else if (strcmp(name, "cloglog") == 0) {
        layers[length-1].activation = cloglog;
        layers[length-1].activation_deriv = cloglog_deriv;
    }
    else if (strcmp(name, "softplus") == 0) {
        layers[length-1].activation = softplus;
        layers[length-1].activation_deriv = softplus_deriv;
    }
    else if (strcmp(name, "relu") == 0) {
        layers[length-1].activation = rectifier(linear);
        layers[length-1].activation_deriv = rectifier(linear_deriv);
    }
    else if (strcmp(name, "leaky_relu") == 0) {
        layers[length-1].activation = leaky_relu;
        layers[length-1].activation_deriv = leaky_relu_deriv;
    }
    else if (strcmp(name, "bipolar_sigmoid") == 0) {
        layers[length-1].activation = bipolar_sigmoid;
        layers[length-1].activation_deriv = bipolar_sigmoid_deriv;
    }
    else if (strcmp(name, "tanh") == 0) {
        layers[length-1].activation = [](float x) -> float {return tanh(x);};
        layers[length-1].activation_deriv = [](float x) -> float {return 1.0/cosh(x);};
    }
    else if (strcmp(name, "hard_tanh") == 0) {
        layers[length-1].activation = hard_tanh;
        layers[length-1].activation_deriv = hard_tanh_deriv;
    }
    else if (strcmp(name, "resig") == 0) {
        layers[length-1].activation = rectifier(sigmoid);
        layers[length-1].activation_deriv = rectifier(sigmoid_deriv);
    }
    else {
        std::cout << "Warning! Incorrect activation specified. Exiting...\n\nIf this is coming up and you don't know why, try defining your own activation function.\n";
        exit(1);
    }
}

void Network::initialize()
{
    labels = new Eigen::MatrixXf (batch_size,layers[length-1].contents->cols());
    for (int i = 0; i < length-1; i++) {
        layers[i].init_weights(layers[i+1]);
    }
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
#pragma omp simd
        for (int k = 0; k < layers[length-1].contents->cols(); k++) {
            (*layers[length-1].dZ)(j,k) = layers[length-1].activation_deriv((*layers[length-1].contents)(j,k));
            (*layers[length-1].contents)(j,k) = layers[length-1].activation((*layers[length-1].contents)(j,k));
        }
    }
    //  std::cout << "\nSOFTMAX INPUT\n" << *layers[length-1].contents << "\n\n";
    for (int i = 0; i < layers[length-1].contents->rows(); i++) {
        float sum = 0;
        Eigen::MatrixXf m = layers[length-1].contents->block(i,0,1,layers[length-1].contents->cols());
        Eigen::MatrixXf::Index maxRow, maxCol;
        float max = m.maxCoeff(&maxRow, &maxCol);
        m = (m.array() - max).matrix();
        // std::cout << "\nGETTING SUM\n";
        for (int j = 0; j < layers[length-1].contents->cols(); j++) {
            checknan(m(0,j), "input to final layer");
            sum += exp(m(0,j));.
            // std::cout << "Adding " << exp(m(0,j)) << "(aka e^"<< m(0, j) << ")\n";
            checknan(sum, "sum in Softmax operation");
        }
        //std::cout << "\nFINAL ACTIVATION\n";
        for (int j = 0; j < layers[length-1].contents->cols(); j++) {
            m(0,j) = exp(m(0,j))/sum;
            //      std::cout << "Calculating " << exp(m(0,j)) << "/" << sum << " to be " << (*layers[length-1].contents)(i,j) << "(aka " << test<<")\n";
            checknan(m(0,j), "output of Softmax operation");
        }
        layers[length-1].contents->block(i,0,1,layers[length-1].contents->cols()) = m;
    }
    // std::cout << "\n\n";
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
        if (reg_type == 2) reg += avx_product(*layers[i].weights,*layers[i].weights).sum();
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

Eigen::MatrixXf Network::numerical_grad(int i, float epsilon)
{
    Eigen::MatrixXf gradient (layers[i].weights->rows(), layers[i].weights->cols());
    for (int j = 0; j < layers[i].weights->rows(); j++) {
        for (int k = 0; k < layers[i].weights->cols(); k++) {
            float current_cost = cost();
            std::vector<Layer> backup = layers;
            (*layers[i].weights)(j,k) += epsilon;
            feedforward();
            float end_cost = cost();
            gradient(j,k) = (end_cost - current_cost)/epsilon;
            layers = backup;
            batches = 0;
        }
    }
    return gradient;
}

void Network::grad_check()                      \
{
    std::vector<Layer> backup = layers;
    feedforward();
    layers = backup;
    batches = 0;
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
    int counter = 1;
    gradients.push_back(error);
    deltas.push_back((*layers[length-2].contents).transpose() * gradients[0]);
    for (int i = length-2; i >= 1; i--) {
        gradients.push_back(avx_product(gradients[counter-1] * layers[i].weights->transpose(),*layers[i].dZ));
        std::cout << layers[i-1].contents->transpose() * gradients[counter];
        deltas.push_back(layers[i-1].contents->transpose() * gradients[counter]);
        counter++;
    }
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
        // TODO: Find nice way to add this
        // (*layers[i].weights-((learning_rate * *layers[i].weights) + (0.9 * *layers[i].v))).transpose()
        grad_calc(gradients, counter, i);
        deltas.push_back(layers[i-1].contents->transpose() * gradients[counter]);
        counter++;
    }
    //  std::cout << deltas[0] << "\n\nNUMERIC\n\n" << numerical_grad(1, 0.000001) <<"\n\n\n-----------\n\n\n";
    for (int i = 0; i < length-1; i++) {
        update(deltas, i);
        //    *layers[length-2-i].weights -= (0.9 * *layers[length-2-i].v) + (learning_rate * deltas[i]);
        //    *layers[length-2-i].v = (learning_rate * deltas[i]);

        if (reg_type == 2) *layers[length-2-i].weights -= ((lambda/batch_size) * (*layers[length-2-i].weights));
        else if (reg_type == 1) *layers[length-2-i].weights -= ((lambda/(2*batch_size)) * l1_deriv(*layers[length-2-i].weights));

        *layers[length-1-i].bias -= bias_lr * gradients[i];
        //    std::cout << *layers[length-2-i].v << "\n\n" << *layers[length-2-i].weights << "\n\n" << deltas[i] << "\n\n\n\n";
    
        if (strcmp(layers[length-2-i].activation_str, "prelu") == 0) {
            float sum = 0;
            for (int j = 0; j < layers[length-2-i].contents->rows(); j++) {
                for (int k = 0; k < layers[length-2-i].contents->cols(); k++) {
                    if ((*layers[length-2-i].contents)(j,k)/layers[length-2-i].alpha <= 0) {
                        // Choice of using index i+1 here is questionable. TODO: REVIEW
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
    for (int i = 0; i < datalen; i++) {
        (*layers[index].contents)((int)i / layers[index].contents->cols(),i%layers[index].contents->cols()) = vals[i];
    }
}

int Network::next_batch()
{
    char line[MAXLINE] = {' '};
    int inputs = layers[0].contents->cols();
    int datalen = batch_size * inputs;
    float batch[datalen];
    for (int i = 0; i < batch_size; i++) {
        fgets(line, MAXLINE, data);
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
    return 0;
}

int prep_file(char* path, char* out_path)
{
    FILE* rptr = fopen(path, "r");
    char line[MAXLINE];
    std::vector<std::string> lines;
    int count = 0;
    while (fgets(line, MAXLINE, rptr) != NULL) {
        lines.emplace_back(line);
        count++;
    }
    lines[lines.size()-1] = lines[lines.size()-1] + "\n";
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(lines.begin(), lines.end(), g);
    fclose(rptr);
    FILE* wptr = fopen(out_path, "w");
    for (std::string & i : lines) {
        const char* cstr = i.c_str();
        fprintf(wptr,"%s", cstr);
    }
    fclose(wptr);
    return count;
}

int split_file(char* path, int lines, float ratio)
{
    FILE* src = fopen(path, "r");
    FILE* test = fopen(VAL_PATH, "w");
    FILE* train = fopen(TRAIN_PATH, "w");
    int switch_line = round(ratio * lines);
    char line[MAXLINE];
    int tests = 0;
    for (int i = 0; fgets(line, MAXLINE, src) != NULL; i++) {
        if (i > switch_line) {
            fprintf(test, "%s", line);
            tests++;
        }
        else fprintf(train, "%s", line);
    }
    fclose(src);
    fclose(test);
    fclose(train);
    return tests;
}

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
    val_acc = 1.0/((float) val_instances/batch_size) * accsum;
    val_cost = 1.0/((float) val_instances/batch_size) * costsum;
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
        if (i != instances-batch_size) { // Don't try to advance batch on final batch.
            next_batch();
        }
        feedforward();
        backpropagate();
        cost_sum += cost();
        acc_sum += accuracy();
        batches++;
        // if (i > batch_size * 10) {
        //   exit(1);
        // }
    }
    epoch_acc = 1.0/((float) instances/batch_size) * acc_sum;
    epoch_cost = 1.0/((float) instances/batch_size) * cost_sum;
    validate(VAL_PATH);
    printf("Epoch %i complete - cost %f - acc %f - val_cost %f - val_acc %f\n", epochs, epoch_cost, epoch_acc, val_cost, val_acc);
    batches=1;
    rewind(data);
    decay();
    epochs++;
}

float Network::get_acc() {return epoch_acc;}
float Network::get_val_acc() {return val_acc;}
float Network::get_cost() {return epoch_cost;}
float Network::get_val_cost() {return val_cost;}
