#include "bpnn.hpp"
#include "utils.hpp"
#include <ctime>
#include <random>

#define SHUFFLED_PATH "./shuffled.txt"
#define TEST_PATH "./test.txt"
#define TRAIN_PATH "./train.txt"

#define MAXLINE 1024
#define ZERO_THRESHOLD pow(10, -8) // for checks

Layer::Layer(int batch_sz, int nodes)
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
  weights = new Eigen::MatrixXf (contents->cols(), next.contents->cols());
  v = new Eigen::MatrixXf (contents->cols(), next.contents->cols());
  int nodes = weights->cols();
  int n = contents->cols() + next.contents->cols();
  std::normal_distribution<float> d(0,sqrt(1.0/n));
  for (int i = 0; i < (weights->rows()*weights->cols()); i++) {
    std::random_device rd;
    std::mt19937 gen(rd()); 
    (*weights)((int)i / nodes, i%nodes) = d(gen);
  }
  for (int i = 0; i < (weights->rows()*weights->cols()); i++) {
    (*v)((int)i / nodes, i%nodes) = 0;
  }
}

Network::Network(char* path, int batch_sz, float learn_rate, float bias_rate, float l, float ratio)
{
  lambda = l;
  learning_rate = learn_rate;
  bias_lr = bias_rate;
  int total_instances = prep_file(path, SHUFFLED_PATH);
  test_instances = split_file(SHUFFLED_PATH, total_instances, ratio);
  instances = total_instances - test_instances;
  batch_size = batch_sz;
  data = fopen(TRAIN_PATH, "r");
  decay = [](float lr, float t) -> float {
    return lr;
  };
}

void Network::init_decay(char* type, float a_0, float k)
{
  if (strcmp(type, "step") == 0) {
    decay = [a_0, k](float lr, float t) -> float {
      return lr/k;
    };
  }
  if (strcmp(type, "exp") == 0) {
    decay = [a_0, k](float lr, float t) -> float {
      return a_0 * exp(-k*t);
    };
  }
  if (strcmp(type, "frac") == 0) {
    decay = [a_0, k](float lr, float t) -> float {
      return a_0/(1+(k*t));
    };
  }
}

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
  for (int j = 0; j < layers[0].contents->rows(); j++) {
    for (int k = 0; k < layers[0].contents->cols(); k++) {
      (*layers[0].dZ)(j,k) = layers[0].activation_deriv((*layers[0].contents)(j,k));
      (*layers[0].contents)(j,k) = layers[0].activation((*layers[0].contents)(j,k));
    }
  }
  for (int i = 0; i < length-1; i++) {
    //if (batch_size > 64 && batch_size % 4 == 0) {
    //  *layers[i+1].contents = strassen_mul((*layers[i].contents),(*layers[i].weights));
    //}
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
  //std::cout << (*layers[length-1].contents) << "\n";
  for (int i = 0; i < layers[length-1].contents->rows(); i++) {
    float sum = 0;
    Eigen::MatrixXf m = layers[length-1].contents->block(i,0,1,layers[length-1].contents->cols());
    Eigen::MatrixXf::Index maxRow, maxCol;
    float max = m.maxCoeff(&maxRow, &maxCol);
    //    std::cout << m << "(before with max "<< max <<")\n";
    m = (m.array() - max).matrix();
    //    std::cout << m << "(after with max "<< max <<")\n";
    for (int j = 0; j < layers[length-1].contents->cols(); j++) {
      sum += exp(m(0,j));
    }
    for (int j = 0; j < layers[length-1].contents->cols(); j++) {
      //      std::cout << "(e^" << m(0,j) << ")/" << sum << " -> " << exp(m(0,j)) << "/" << sum << " -> " << exp((m(0,j)))/sum << "\n";
      m(0,j) = exp(m(0,j))/sum;
    }
    layers[length-1].contents->block(i,0,1,layers[length-1].contents->cols()) = m;
  }
}

void Network::list_net()
{
  std::cout << "-----------------------\nINPUT LAYER (LAYER 0)\n-----------------------\n\n\u001b[31mGENERAL INFO:\x1B[0;37m\nActivation Function: " << layers[0].activation_str << "\n\n\u001b[31mACTIVATIONS:\x1B[0;37m\n" << *layers[0].contents << "\n\n\u001b[31mWEIGHTS:\x1B[0;37m\n" << *layers[0].weights << "\n\n\u001b[31mBIASES:\x1B[0;37m\n" << *layers[0].bias << "\n\n\n";
  for (int i = 1; i < length-1; i++) {
    std::cout << "-----------------------\nLAYER " << i << "\n-----------------------\n\n\u001b[31mGENERAL INFO:\x1B[0;37m\nActivation Function: " << layers[i].activation_str << "\n\n\u001b[31mACTIVATIONS:\x1B[0;37m\n" << *layers[i].contents << "\n\n\u001b[31mBIASES:\x1B[0;37m\n" << *layers[i].bias << "\n\n\u001b[31mWEIGHTS:\x1B[0;37m\n" << *layers[i].weights << "\n\n\n";
  }
  std::cout << "-----------------------\nOUTPUT LAYER (LAYER " << length-1 << ")\n-----------------------\n\n\u001b[31mGENERAL INFO:\x1B[0;37m\nActivation Function: " << layers[length-1].activation_str <<"\n\n\u001b[31mACTIVATIONS:\x1B[0;37m\n" << *layers[length-1].contents << "\n\n\u001b[31mBIASES:\x1B[0;37m\n" << *layers[length-1].bias <<  "\n\n\n";
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
      //      std::cout << truth << " VS " << (*layers[length-1].contents)(i,j) << " SO " << truth * log((*layers[length-1].contents)(i,j)) << "\n";
      tempsum += truth * log((*layers[length-1].contents)(i,j));
    }
    sum-=tempsum;
  }
  for (int i = 0; i < layers.size()-1; i++) {
    reg += (layers[i].weights->cwiseProduct(*layers[i].weights)).sum();
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
        //std::cout << "UPDATE ANS: " << index << " as "<< (*layers[length-1].contents)(i, j) << " so " << ans << "\n";
      }
    }
    //std::cout << (*labels)(i, 0) << " " << index << "\n";
    if ((*labels)(i, 0) == index) correct += 1;
  }
  return (1.0/batch_size) * correct;
}

void Network::backpropagate()
{
  std::vector<Eigen::MatrixXf> gradients;
  std::vector<Eigen::MatrixXf> deltas;
  Eigen::MatrixXf error (layers[length-1].contents->rows(), layers[length-1].contents->cols());
  // std::cout << (*layers[length-1].contents) << "\n\n\n";
  for (int i = 0; i < error.rows(); i++) {
    for (int j = 0; j < error.cols(); j++) {
      float truth;
      if (j==(*labels)(i,0)) truth = 1;
      else truth = 0;
      error(i,j) = truth - (*layers[length-1].contents)(i,j);
      // std::cout << truth << "[as label is "<< (*labels)(i,0) <<"] - " << (*layers[length-1].contents)(i,j) << "[aka index " << i << " " << j << "] = " << error(i,j) << "\n";
    }
  }
  //  std::cout << error << "\n\n";
  gradients.push_back(error);
  deltas.push_back((*layers[length-2].contents).transpose() * gradients[0]);
  //  std::cout << deltas[0] << "\n\n";
  //  gradients[523] += error;
  int counter = 1;
  for (int i = length-2; i >= 1; i--) {
    gradients.push_back((gradients[counter-1] * layers[i].weights->transpose()).cwiseProduct(*layers[i].dZ));
    deltas.push_back(layers[i-1].contents->transpose() * gradients[counter]);
    counter++;
  }
  // std::cout << "-------\nGRADS INCOMING" << "\n\n";
  // for (Eigen::MatrixXf i : gradients) {
  //   std::cout << i << "\n\n";
  // }
  // std::cout << "-------\nDELTAS INCOMING" << "\n\n";
  // for (Eigen::MatrixXf i : deltas) {
  //   std::cout << i << "\n\n";
  // }
  for (int i = 0; i < length-1; i++) {
    *layers[length-2-i].weights -= (learning_rate * deltas[i]) + ((lambda/batch_size) * (*layers[length-2-i].weights));
    //*layers[length-2-i].v = (0.9 * *layers[length-2-i].v) - ((learning_rate * deltas[i]));
    //*layers[length-2-i].weights += *layers[length-2-i].v;
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
      fgets(line, MAXLINE, test_data);
      //if (strcmp(line, "\n")==0) continue;
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

#include "checks.cpp"

void Network::train()
{
  rewind(data);
  float cost_sum = 0;
  float acc_sum = 0;
  for (int i = 0; i <= instances-batch_size; i+=batch_size) {
    [[unlikely]] if (i != instances-batch_size) { // Don't try to advance batch on final batch.
      next_batch();
    }
    feedforward();
    backpropagate();
    cost_sum += cost();
    acc_sum += accuracy();
    batches++;
    // if (i > batch_size * 10) {
    //   list_net();
    //   exit(1);
    // }
  }
  epoch_acc = 1.0/((float) instances/batch_size) * acc_sum;
  epoch_cost = 1.0/((float) instances/batch_size) * cost_sum;
  test(TEST_PATH);
  printf("Epoch %i complete - cost %f - acc %f - val_cost %f - val_acc %f\n", epochs, epoch_cost, epoch_acc, val_cost, val_acc);
  batches=1;
  rewind(data);
  learning_rate = decay(learning_rate, epochs);
  epochs++;
}

float Network::get_acc() {return epoch_acc;}
float Network::get_val_acc() {return val_acc;}
float Network::get_cost() {return epoch_cost;}
float Network::get_val_cost() {return val_cost;}
