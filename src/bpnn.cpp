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
#include <utility>

#define SHUFFLED_PATH "./shuffled.txt"
#define VAL_PATH "./test.txt"
#define TRAIN_PATH "./train.txt"
#define VAL_BIN_PATH "./test.bin"
#define TRAIN_BIN_PATH "./train.bin"
#define VAL_LZ4_PATH "./test.lz4"
#define TRAIN_LZ4_PATH "./train.lz4"

Layer::Layer(int batch_sz, int nodes)
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

Network::Network(const char* path, int batch_sz, float learn_rate, float bias_rate, Regularization regularization, float l, float ratio, bool early_exit, float cutoff)
    :batch_size(batch_sz), learning_rate(learn_rate), bias_lr(bias_rate), reg_type(regularization), lambda(l), early_stop(early_exit), threshold(cutoff)
{
    Expects(batch_size > 0 && learning_rate > 0 &&
            bias_rate > 0 && l >= 0 && ratio >= 0 && ratio <= 1);
    int total_instances = prep_file(path, SHUFFLED_PATH);
    val_instances = split_file(SHUFFLED_PATH, total_instances, ratio);
    prep(TRAIN_PATH, TRAIN_BIN_PATH);
    prep(VAL_PATH, VAL_BIN_PATH);
    data = open(TRAIN_BIN_PATH, O_RDONLY | O_NONBLOCK);
    val_data = open(VAL_BIN_PATH, O_RDONLY | O_NONBLOCK);
    instances = total_instances - val_instances;
    decay = [this]() -> void {};
    update = [this](std::vector<Eigen::MatrixXf> deltas, int i) {
        *layers[length-2-i].weights -= (learning_rate * deltas[i]);
    };
    // File descriptors are nonnegative integers and open() returns -1 on failure. 
    Ensures(batch_size < instances && data > 0 && val_data > 0);
}

Network::~Network()
{
    close(data);
    close(val_data);
}

void Network::add_layer(int nodes, const char* name, std::function<float(float)> activation, std::function<float(float)> activation_deriv)
{
    Expects(nodes > 0);
    length++;
    layers.emplace_back(batch_size, nodes);
    strcpy(layers[length-1].activation_str, name);
    layers[length-1].activation = activation;
    layers[length-1].activation_deriv = activation_deriv;
}

void Network::initialize()
{
    Expects(length > 1);
    for (int i = 0; i < length-1; i++) layers[i].init_weights(layers[i+1]);
}

void Network::set_activation(int index, std::function<float(float)> custom, std::function<float(float)> custom_deriv)
{
    Expects(index >= 0 && index < length);
    layers[index].activation = custom;
    layers[index].activation_deriv = custom_deriv;
}

Eigen::MatrixXf Network::softmax(Eigen::MatrixXf matrix)
{
    for (int i = 0; i < matrix.rows(); i++) {
        Eigen::MatrixXf m = matrix.block(i,0,1,matrix.cols());
        Eigen::MatrixXf::Index maxRow, maxCol;
        float max = m.maxCoeff(&maxRow, &maxCol);
        m = (m.array() - max).matrix();
        float sum = 0;
        for (int j = 0; j < matrix.cols(); j++) {
            checknan(m(0,j), "input of Softmax operation");
        for (int j = 0; j < layers[length-1].contents->cols(); j++) {
            sum += exp(m(0,j));
        }
        for (int j = 0; j < matrix.cols(); j++) {
            m(0,j) = exp(m(0,j))/sum;
        }
#endif
        matrix.block(i,0,1,matrix.cols()) = m;
    }
    return matrix;
}

std::pair<std::vector<Eigen::MatrixXf>, std::vector<Eigen::MatrixXf>> Network::virtual_feedforward(Eigen::MatrixXf init)
{
    Eigen::MatrixXf out (layers[length-1].contents->rows(), layers[length-1].contents->cols());
    std::vector<Eigen::MatrixXf> dZ;
    std::vector<Eigen::MatrixXf> virt_layers;
    dZ.emplace_back(layers[0].dZ->rows(), layers[0].dZ->cols());
    for (int j = 0; j < layers[0].contents->rows(); j++) {
        if (strcmp(layers[0].activation_str, "linear") == 0) break;
        for (int k = 0; k < layers[0].contents->cols(); k++) {
            dZ[0](j,k) = layers[0].activation_deriv(init(j,k));
            init(j,k) = layers[0].activation(init(j,k));
        }
    }
    out = init;
    virt_layers.push_back(out);
    for (int i = 1; i < length; i++) {
        dZ.emplace_back(layers[i].dZ->rows(), layers[i].dZ->cols());
        out = out * (*layers[i-1].weights);
        for (int j = 0; j < layers[i].contents->rows(); j++) {
            if (strcmp(layers[i].activation_str, "linear") == 0) break;
            for (int k = 0; k < layers[i].contents->cols(); k++) {
                dZ[i](j,k) = layers[i].activation_deriv(out(j,k));
                out(j,k) = layers[i].activation(out(j,k));
            }
        }
        virt_layers.push_back(out);
    }
    out = softmax(out);      
    virt_layers[virt_layers.size()-1] = out;
    return {virt_layers, dZ};
}

void Network::list_net()
{
    Expects(length > 1);
    std::cout << "-----------------------\nINPUT LAYER (LAYER 0)\n-----------------------\n\n\u001b[31mGENERAL INFO:\x1B[0;37m\nActivation Function: " << layers[0].activation_str << "\n\n\u001b[31mACTIVATIONS:\x1B[0;37m\n" << *layers[0].contents << "\n\n\u001b[31mWEIGHTS:\x1B[0;37m\n" << *layers[0].weights << "\n\n\u001b[31mBIASES:\x1B[0;37m\n" << *layers[0].bias << "\n\n\n";
    for (int i = 1; i < length-1; i++) {
        std::cout << "-----------------------\nLAYER " << i << "\n-----------------------\n\n\u001b[31mGENERAL INFO:\x1B[0;37m\nActivation Function: " << layers[i].activation_str;
        std::cout << "\n\n\u001b[31mACTIVATIONS:\x1B[0;37m\n" << *layers[i].contents << "\n\n\u001b[31mBIASES:\x1B[0;37m\n" << *layers[i].bias << "\n\n\u001b[31mWEIGHTS:\x1B[0;37m\n" << *layers[i].weights << "\n\n\n";
    }
    std::cout << "-----------------------\nOUTPUT LAYER (LAYER " << length-1 << ")\n-----------------------\n\n\u001b[31mGENERAL INFO:\x1B[0;37m\nActivation Function: " << layers[length-1].activation_str <<"\n\n\u001b[31mACTIVATIONS:\x1B[0;37m\n" << *layers[length-1].contents << "\n\n\u001b[31BIASES:\x1B[0;37m\n" << *layers[length-1].bias <<  "\n\n\n";
}

float Network::cost(Eigen::MatrixXf labels, Eigen::MatrixXf out)
{
    float sum = 0;
    float reg = 0; // Regularization term
    for (int i = 0; i < out.rows(); i++) {
        float tempsum = 0;
        for (int j = 0; j < out.cols(); j++) {
            float truth;
            if (j==labels(i,0)) truth = 1;
            else truth = 0;
            if ((*layers[length-1].contents)(i,j) == 0) (*layers[length-1].contents)(i,j) += 0.00001;
            tempsum += truth * log((*layers[length-1].contents)(i,j));
        }
        sum-=tempsum;
    }
    for(unsigned long i = 0; i < layers.size()-1; i++) {
        if (reg_type == L2) reg += layers[i].weights->cwiseProduct(*layers[i].weights).sum();
        else if (reg_type == L1) reg += (layers[i].weights->array().abs().matrix()).sum();
    }
    return ((1.0/batch_size) * sum) + (1/2*lambda*reg);
}

float Network::accuracy(Eigen::MatrixXf labels, Eigen::MatrixXf out)
{
    float correct = 0;
    for (int i = 0; i < out.rows(); i++) {
        float ans = -INFINITY;
        float index = -1;
        for (int j = 0; j < out.cols(); j++) {
            if (out(i, j) > ans) {
                ans = out(i, j);
                index = j;
            }
        }
        if (labels(i, 0) == index) correct += 1;
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

void Network::virtual_backprop(Eigen::MatrixXf labels, std::vector<Eigen::MatrixXf> virt_layers, std::vector<Eigen::MatrixXf> dZ)
{
    std::vector<Eigen::MatrixXf> gradients;
    std::vector<Eigen::MatrixXf> deltas;
    Eigen::MatrixXf error (virt_layers[length-1].rows(), virt_layers[length-1].cols());
    for (int i = 0; i < error.rows(); i++) {
        for (int j = 0; j < error.cols(); j++) {
            float truth;
            if (j==labels(i,0)) truth = 1;
            else truth = 0;
            error(i,j) = virt_layers[length-1](i,j) - truth;
            checknan(error(i,j), "gradient of final layer");
        }
    }
    gradients.push_back(error);
    deltas.push_back(virt_layers[length-2].transpose() * gradients[0]);
    int counter = 1;
    for (int i = length-2; i >= 1; i--) {
        // TODO: Add nesterov momentum | -p B -t conundrum -t coding -m Without causing segmentation faults.        
        // (*layers[i].weights-((learning_rate * *layers[i].weights) + (0.9 * *layers[i].v))).
        //grad_calc(gradients, counter, i)]
        gradients.push_back((gradients[counter-1] * layers[i].weights->transpose()).cwiseProduct(dZ[i]));
        deltas.push_back(virt_layers[i-1].transpose() * gradients[counter]);
        counter++;
    }
    for (int i = 0; i < length-1; i++) {
        update(deltas, i);
        if (reg_type == L2) *layers[length-2-i].weights -= ((lambda/batch_size) * (*layers[length-2-i].weights));
        else if (reg_type == L1) *layers[length-2-i].weights -= ((lambda/(2*batch_size)) * l1_deriv(*layers[length-2-i].weights));

        *layers[length-1-i].bias -= bias_lr * gradients[i];
    }
    return gradients.back();
}

#include "data.cpp"

void Network::validate(const char* path)
{
    if (val_instances == 0) return;
    float costsum = 0;
    float accsum = 0;
    for (int i = 0; i <= val_instances-batch_size; i+=batch_size) {
        std::pair<Eigen::MatrixXf,Eigen::MatrixXf> batch = next_batch(val_data);
        std::pair<std::vector<Eigen::MatrixXf>, std::vector<Eigen::MatrixXf>> vals = virtual_feedforward(batch.first);        
        costsum += cost(batch.second, vals.first[vals.first.size()-1]);
        accsum += accuracy(batch.second, vals.first[vals.first.size()-1]);
    }
    val_acc = 1.0/(static_cast<float>(val_instances/batch_size)) * accsum;
    val_cost = 1.0/(static_cast<float>(val_instances/batch_size)) * costsum;
    val_data = open(VAL_BIN_PATH, O_RDONLY | O_NONBLOCK);
    Ensures(lseek(val_data, 0, SEEK_CUR) == 0);
}

void Network::run(std::vector<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>> batches)
{
    for (std::pair<Eigen::MatrixXf, Eigen::MatrixXf> batch : batches) {
        std::pair<std::vector<Eigen::MatrixXf>, std::vector<Eigen::MatrixXf>> vals = virtual_feedforward(batch.first);
        virtual_backprop(batch.second, vals.first, vals.second);
    }
}

#ifdef __APPLE__

typedef struct cpu_set {
  uint32_t    count;
} cpu_set_t;

// This person is my savior
// http://www.hybridkernel.com/2015/01/18/binding_threads_to_cores_osx.html

static inline void
CPU_ZERO(cpu_set_t *cs) { cs->count = 0; }

static inline void
CPU_SET(int num, cpu_set_t *cs) { cs->count |= (1 << num); }

static inline int CPU_ISSET(int num, cpu_set_t *cs) { return (cs->count & (1 << num)); }

int pthread_setaffinity_np(pthread_t thread, size_t cpu_size,
                           cpu_set_t *cpu_set)
{
  thread_port_t mach_thread;
  int core = 0;

  for (core = 0; core < 8 * cpu_size; core++) {
    if (CPU_ISSET(core, cpu_set)) break;
  }
  //printf("binding to core %d\n", core);
  thread_affinity_policy_data_t policy = { core };
  mach_thread = pthread_mach_thread_np(thread);
  thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY,
                    (thread_policy_t)&policy, 1);
  return 0;
}

#endif

void Network::train()
{
    float cost_sum = 0;
    float acc_sum = 0;
    unsigned int cores = std::thread::hardware_concurrency();
    std::vector<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>> info;
    for (int i = 0; i <= instances-batch_size; i+=batch_size*cores) {
        info.push_back(next_batch(data));
    }
    std::vector<std::vector<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>>> bunches;
    // https://stackoverflow.com/questions/40656792/c-best-way-to-split-vector-into-n-vector
    int bunch_size = info.size() / cores;
    for(size_t i = 0; i < info.size(); i += bunch_size) {
        auto last = std::min(info.size(), i + bunch_size);
        bunches.emplace_back(info.begin() + i, info.begin() + last);
    }    
    if (early_stop == true && get_val_cost() < threshold) return;
    std::vector<std::thread> threads;
    for (int i = 0; i < cores; i++) {
        threads.emplace_back(&Network::run, this, bunches[i]);
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(i, &cpuset);
        int rc = pthread_setaffinity_np(threads[i].native_handle(),
                                        sizeof(cpu_set_t), &cpuset);
        if (rc != 0) {
            std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
        }
    }
    for (int i = 0; i < cores; i++) threads[i].join();
    //cost_sum += cost();
    //acc_sum += accuracy();   
    epoch_acc = 1.0/(static_cast<float>(instances/batch_size)) * acc_sum;
    epoch_cost = 1.0/(static_cast<float>(instances/batch_size)) * cost_sum;
    validate(VAL_PATH);
    if (silenced == false) printf("Epoch %i complete - cost %f - acc %f - val_cost %f - val_acc %f\n", epochs, epoch_cost, epoch_acc, val_cost, val_acc);
    batches=1;
    data = open(TRAIN_BIN_PATH, O_RDONLY | O_NONBLOCK);
    decay();
    epochs++;
    Ensures(lseek(data, 0, SEEK_CUR) == 0);
}

