#include "bpnn.hpp"
#include "utils.hpp"
#include <ctime>
#include <random>

Layer::Layer(float* vals, int batch_sz, int nodes)
{
  contents = new Eigen::MatrixXd (batch_sz, nodes);
  dZ = new Eigen::MatrixXd (batch_sz, nodes);
  int datalen = batch_sz*nodes;
  for (int i = 0; i < datalen; i++) {
    (*contents)((int)i / nodes,i%nodes) = vals[i];
    (*dZ)((int)i / nodes,i%nodes) = 0;
  }
  bias = new Eigen::MatrixXd (batch_sz, nodes);
  for (int i = 0; i < nodes; i++) {
    for (int j = 0; j < batch_sz; j++) {
      (*bias)(j, i) = 0.001;
    }
  }
}

Layer::Layer(int batch_sz, int nodes)
{
  contents = new Eigen::MatrixXd (batch_sz, nodes);
  dZ = new Eigen::MatrixXd (batch_sz, nodes);
  int datalen = batch_sz*nodes;
  for (int i = 0; i < datalen; i++) {
    (*contents)((int)i / nodes,i%nodes) = 0;
    (*dZ)((int)i / nodes,i%nodes) = 0;
  }
  bias = new Eigen::MatrixXd (batch_sz, nodes);
  for (int i = 0; i < nodes; i++) {
    for (int j = 0; j < batch_sz; j++) {
      (*bias)(j, i) = 0.001;
    }
  }
  dZ = new Eigen::MatrixXd (batch_sz, nodes);
}

void Layer::init_weights(Layer next)
{
  weights = new Eigen::MatrixXd (contents->cols(), next.contents->cols());
  int nodes = weights->cols();
  int n = contents->cols() + next.contents->cols();
  std::normal_distribution<float> d(0,sqrt(1.0/n));
  for (int i = 0; i < (weights->rows()*weights->cols()); i++) {
    std::random_device rd;
    std::mt19937 gen(rd()); 
    (*weights)((int)i / nodes, i%nodes) = d(gen);
  }
}

Network::Network(char* path, int batch_sz, float learn_rate, float bias_rate)
{
  learning_rate = learn_rate;
  bias_lr = bias_rate;
  instances = prep_file(path, "./shuffled.txt");
  length = 0;
  batch_size = batch_sz;
  data = fopen("./shuffled.txt", "r");
  batches = 0;
}

void Network::add_layer(int nodes, char* activation)
{
  length++;
  layers.emplace_back(batch_size, nodes);
  set_activation(length-1, activation);
}

void Network::initialize()
{
  labels = new Eigen::MatrixXd (batch_size,layers[length-1].contents->cols());
  for (int i = 0; i < length-1; i++) {
    layers[i].init_weights(layers[i+1]);
  }
}

void Network::set_activation(int index, char* name)
{
  if (strcmp(name, "sigmoid") == 0) {
    layers[index].activation = &sigmoid;
    layers[index].activation_deriv = &sigmoid_deriv;
  }
  else if (strcmp(name, "linear") == 0) {
    layers[index].activation = &linear;
    layers[index].activation_deriv = &linear_deriv;
  }
  else if (strcmp(name, "relu") == 0) {
    layers[index].activation = &relu;
    layers[index].activation_deriv = &relu_deriv;
  }
  else if (strcmp(name, "resig") == 0) {
    layers[index].activation = &resig;
    layers[index].activation_deriv = &resig_deriv;
  }
  else {
    std::cout << "Warning! Incorrect activation specified. Exiting...\n";
    exit(1);
  }
}

void Network::feedforward()
{
  for (int j = 0; j < layers[0].contents->rows(); j++) {
    for (int k = 0; k < layers[0].contents->cols(); k++) {
      (*layers[0].dZ)(j,k) = (*layers[0].activation_deriv)((*layers[0].contents)(j,k));
      (*layers[0].contents)(j,k) = (*layers[0].activation)((*layers[0].contents)(j,k));
    }
  }
  for (int i = 0; i < length-1; i++) {
    *layers[i+1].contents = (*layers[i].contents) * (*layers[i].weights);
    *layers[i+1].contents += *layers[i+1].bias;
  }
  for (int i = 1; i < length; i++) {
    for (int j = 0; j < layers[i].contents->rows(); j++) {
      for (int k = 0; k < layers[i].contents->cols(); k++) {
        (*layers[i].dZ)(j,k) = (*layers[i].activation_deriv)((*layers[i].contents)(j,k));
        (*layers[i].contents)(j,k) = (*layers[i].activation)((*layers[i].contents)(j,k));
      }
    }
  }
}

void Network::list_net()
{
  for (int i = 0; i < length-1; i++) {
    std::cout << " LAYER " << i << "\n\n" << *layers[i].contents << "\n\n AND BIAS\n" << *layers[i].bias << "\n\n W/ WEIGHTS \n" << *layers[i].weights << "\n\n\n";
  }
  std::cout << " LAYER " << length-1 << "\n\n" << *layers[length-1].contents << "\n\n AND BIAS\n" << *layers[length-1].bias <<  "\n\n\n";
}

float Network::cost()
{
  float sum = 0;
  for (int i = 0; i < layers[length-1].contents->rows(); i++) {
    sum += pow((*labels)(i, 0) - (*layers[length-1].contents)(i, 0),2);
  }
  return (1.0/batch_size) * sum;
}

float Network::accuracy()
{
  float correct = 0;
  int total = 0;
  for (int i = 0; i < layers[length-1].contents->rows(); i++) {
    if ((*labels)(i, 0) == round((*layers[length-1].contents)(i, 0))) {
      correct += 1;
    }
    total = i;
  }
  return (1.0/batch_size) * correct;
}

void Network::backpropagate()
{
  std::vector<Eigen::MatrixXd> gradients;
  std::vector<Eigen::MatrixXd> deltas;
  Eigen::MatrixXd error = ((*layers[length-1].contents) - (*labels));
  gradients.push_back(error.cwiseProduct(*layers[length-1].dZ));
  deltas.push_back((*layers[length-2].contents).transpose() * gradients[0]);
  int counter = 1;
  for (int i = length-2; i >= 1; i--) {
    gradients.push_back((gradients[counter-1] * layers[i].weights->transpose()).cwiseProduct(*layers[i].dZ));
    deltas.push_back(layers[i-1].contents->transpose() * gradients[counter]);
    counter++;
  }
  for (int i = 0; i < length-1; i++) {
    Eigen::MatrixXd gradient = gradients[i];
    *layers[length-2-i].weights -= learning_rate * deltas[i];
    *layers[length-1-i].bias -= bias_lr * gradients[i];
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
  auto init_begin = std::chrono::high_resolution_clock::now();
  char line[1024] = {' '};
  int inputs = layers[0].contents->cols();
  int datalen = batch_size * inputs;
  float batch[datalen];
  int label = -1;
  auto get_begin = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < batch_size; i++) {
    if (fgets(line, 1024, data)==NULL) {
      break;
    }
    sscanf(line, "%f,%f,%f,%f,%i", &batch[0 + (i * inputs)],
           &batch[1 + (i * inputs)], &batch[2 + (i * inputs)],
           &batch[3 + (i * inputs)], &label);
    (*labels)(i, 0) = label;
  }
  auto get_end = std::chrono::high_resolution_clock::now();
  float* batchptr = batch;
  update_layer(batchptr, datalen, 0);
  auto update_end = std::chrono::high_resolution_clock::now();
  //std::cout << " INIT " << std::chrono::duration_cast<std::chrono::nanoseconds>(get_begin - init_begin).count() / pow(10,9) << " GET " << std::chrono::duration_cast<std::chrono::nanoseconds>(get_end - get_begin).count() / pow(10,9) << " UPDATE " << std::chrono::duration_cast<std::chrono::nanoseconds>(update_end - get_end).count() / pow(10,9) << " TOTAL " << std::chrono::duration_cast<std::chrono::nanoseconds>(update_end - init_begin).count() / pow(10,9) << "\n";
  //  std::cout << "Next batch is\n" << *layers[0].contents << "\nwith labels\n"<<*labels << "\n\n";
  return 0;
}

int prep_file(char* path, char* out_path)
{
  FILE* rptr = fopen(path, "r");
  char line[1024];
  std::vector<std::string> lines;
  int count = 0;
  while (fgets(line, 1024, rptr) != NULL) {
    lines.emplace_back(line);
    count++;
  }
  lines[lines.size()-1] = lines[lines.size()-1] + "\n";
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(lines.begin(), lines.end(), g);
  fclose(rptr);
  std::ofstream out(out_path);
  for (int i = 0; i < lines.size(); i++) {
    out << lines[i];
  }
  out.close();
  return count;
}

float Network::test(char* path)
{
  int rounds = 1;
  int exit = 0;
  int linecount = prep_file(path, "./testshuffled");
  float cost_sum = 0;
  float acc_sum = 0;
  int finalcount;
  for (int i = 0; i < linecount; i+=batch_size) {
    feedforward();
    FILE* fptr = fopen("./testshuffled", "r");
    char line[1024] = {' '};
    int inputs = layers[0].contents->cols();
    int datalen = batch_size * inputs;
    float batch[datalen];
    int label = -1;
    for (int j = 1; j < batch_size*((i/batch_size)+1); j++) {
      if (fgets(line, 1024, fptr)==NULL) {
        break;
      }
      if (i >= (i/batch_size)*batch_size) {
        int k = i - ((i/batch_size)*batch_size);
        sscanf(line, "%f,%f,%f,%f,%i", &batch[0 + (k * inputs)],
                &batch[1 + (k * inputs)], &batch[2 + (k * inputs)],
                &batch[3 + (k * inputs)], &label);
        (*labels)(k, 0) = label;
      }
    }
    float* batchptr = batch;
    update_layer(batchptr, datalen, 0);
    fclose(fptr);
    cost_sum += cost();
    acc_sum += accuracy();
    finalcount = i;
  }
  float chunks = ((float)finalcount/batch_size)+1;
  return acc_sum/chunks;
}

void Network::train(int total_epochs)
{
  float epoch_cost = 1000;
  float epoch_accuracy = -1;
  int epochs = 0;
  printf("Beginning train on %i instances for %i epochs...\n", instances, total_epochs);
  double batch_time = 0;
  while (epochs < total_epochs) {
    auto ep_begin = std::chrono::high_resolution_clock::now();
    float cost_sum = 0;
    float acc_sum = 0;
    for (int i = 0; i <= instances-batch_size; i+=batch_size) {
      if (i != instances-batch_size) { // Don't try to advance batch on final batch.
        next_batch();
      }
      feedforward();
      backpropagate();
      cost_sum += cost();
      acc_sum += accuracy();
      batches++;
    }
    epoch_accuracy = 1.0/((float) instances/batch_size) * acc_sum;
    epoch_cost = 1.0/((float) instances/batch_size) * cost_sum;
    auto ep_end = std::chrono::high_resolution_clock::now();
    double epochtime = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(ep_end-ep_begin).count() / pow(10,9);
    printf("Epoch %i/%i - time %f - cost %f - acc %f\n", epochs+1, total_epochs, epochtime, epoch_cost, epoch_accuracy);
    batches=1;
    epochs++;
    rewind(data);
  }
}
