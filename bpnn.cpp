#include "bpnn.hpp"
#include "utils.hpp"
#include <ctime>
#include <random>

#define SHUFFLED_PATH "./shuffled.txt"
#define TEST_PATH "./test.txt"
#define TRAIN_PATH "./train.txt"

#define MAXLINE 1024

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
      (*bias)(j, i) = 0;
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
  int total_instances = prep_file(path, SHUFFLED_PATH);
  test_instances = split_file(SHUFFLED_PATH, total_instances, 0.7);
  instances = total_instances - test_instances;
  length = 0;
  t = 0;
  batch_size = batch_sz;
  data = fopen(TRAIN_PATH, "r");
  batches = 0;
}

void Network::add_layer(int nodes, char* name)
{
  length++;
  layers.emplace_back(batch_size, nodes);
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
  labels = new Eigen::MatrixXd (batch_size,layers[length-1].contents->cols());
  for (int i = 0; i < length-1; i++) {
    layers[i].init_weights(layers[i+1]);
  }
}

void Network::set_activation(int index, std::function<double(double)> custom, std::function<double(double)> custom_deriv)
{
  layers[index].activation = custom;
  layers[index].activation_deriv = custom_deriv;
}

void Network::feedforward()
{
  for (int j = 0; j < layers[0].contents->rows(); j++) {
    for (int k = 0; k < layers[0].contents->cols(); k++) {
      (*layers[0].dZ)(j,k) = layers[0].activation_deriv((*layers[0].contents)(j,k));
      (*layers[0].contents)(j,k) = layers[0].activation((*layers[0].contents)(j,k));
    }
  }
  for (int i = 0; i < length-1; i++) {
    *layers[i+1].contents = (*layers[i].contents) * (*layers[i].weights);
    *layers[i+1].contents += *layers[i+1].bias;
  }
  for (int i = 1; i < length; i++) {
    for (int j = 0; j < layers[i].contents->rows(); j++) {
      for (int k = 0; k < layers[i].contents->cols(); k++) {
        (*layers[i].dZ)(j,k) = layers[i].activation_deriv((*layers[i].contents)(j,k));
        (*layers[i].contents)(j,k) = layers[i].activation((*layers[i].contents)(j,k));
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
    sum += ((*labels)(i, 0) - (*layers[length-1].contents)(i, 0)) * ((*labels)(i, 0) - (*layers[length-1].contents)(i, 0));
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
  char line[MAXLINE] = {' '};
  int inputs = layers[0].contents->cols();
  int datalen = batch_size * inputs;
  float batch[datalen];
  int label = -1;
  for (int i = 0; i < batch_size; i++) {
    // This shouldn't ever happen - tell the compiler that.
    fgets(line, MAXLINE, data);
    char *p;
    //    p = strtok (line," ,.-");
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
  std::ofstream out(out_path);
  for (int i = 0; i < lines.size(); i++) {
    out << lines[i];
  }
  out.close();
  return count;
}

int split_file(char* path, int lines, float ratio)
{
  FILE* src = fopen(path, "r");
  FILE* test = fopen(TEST_PATH, "w");
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

float Network::test(char* path)
{
  FILE* test_data = fopen(path, "r");
  float costsum = 0;
  float accsum = 0;
  for (int i = 0; i <= test_instances-batch_size; i+=batch_size) {
    char line[MAXLINE];
    int inputs = layers[0].contents->cols();
    int datalen = batch_size * inputs;
    float batch[datalen];
    int label = -1;
    for (int i = 0; i < batch_size; i++) {
      fgets(line, MAXLINE, data);
      printf("%s", line);
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
  val_acc = 1.0/((float) test_instances/batch_size) * accsum;
  val_cost = 1.0/((float) test_instances/batch_size) * costsum;
  return 0;
}

void Network::train(int total_epochs)
{
  int epochs = 0;
  //printf("Beginning train on %i instances for %i epochs...\n", instances, total_epochs);
  double batch_time = 0;
  while (epochs < total_epochs) {
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
      //t++;
    }
    epoch_acc = 1.0/((float) instances/batch_size) * acc_sum;
    epoch_cost = 1.0/((float) instances/batch_size) * cost_sum;
    test(TEST_PATH);
    printf("Epoch %i/%i - cost %f - acc %f - val_cost %f - val_acc %f\n", epochs+1, total_epochs, epoch_cost, epoch_acc, val_cost, val_acc);
    batches=1;
    epochs++;
    rewind(data);
  }
}

float Network::get_acc()
{
  return epoch_acc;
}

float Network::get_val_acc()
{
  return val_acc;
}

float Network::get_cost()
{
  return epoch_cost;
}

float Network::get_val_cost()
{
  return val_cost;
}

