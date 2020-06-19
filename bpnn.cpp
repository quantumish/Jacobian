#include "bpnn.hpp"
#include <ctime>

Layer::Layer(float* vals, int batch_sz, int nodes)
{
  contents = new Eigen::MatrixXd (batch_sz, nodes);
  int datalen = batch_sz*nodes;
  for (int i = 0; i < datalen; i++) {
    (*contents)((int)i / nodes,i%nodes) = vals[i];
  }
  bias = new Eigen::MatrixXd (1, nodes);
  for (int i = 0; i < nodes; i++) {
    (*bias)(0,i) = 0.001;
  }
}

Layer::Layer(int batch_sz, int nodes)
{
  contents = new Eigen::MatrixXd (batch_sz, nodes);
  int datalen = batch_sz*nodes;
  for (int i = 0; i < datalen; i++) {
    (*contents)((int)i / nodes,i%nodes) = 0;
  }
  bias = new Eigen::MatrixXd (1, nodes);
  for (int i = 0; i < nodes; i++) {
    (*bias)(0,i) = 0.001;
  }
  dZ = new Eigen::MatrixXd (batch_sz, nodes);
}

void Layer::initWeights(Layer next)
{
  weights = new Eigen::MatrixXd (contents->cols(), next.contents->cols());
  int nodes = weights->cols();
  for (int i = 0; i < (weights->rows()*weights->cols()); i++) {
    (*weights)((int)i / nodes, i%nodes) = rand() / double(RAND_MAX);
  }
}

Network::Network(char* path, int inputs, int hidden, int outputs, int neurons, int batch_sz, float rate)
{
  learning_rate = rate;
  fpath = path;
  length = hidden + 2;
  batch_size = batch_sz;
  FILE* fptr = fopen(path, "r");
  int datalen = batch_sz*inputs;
  float batch[datalen];
  labels = new Eigen::MatrixXd (batch_sz, 1);
  int label;
  char line[1024] = {' '};
  for (int i = 0; i < batch_sz; i++) {
    fgets(line, 1024, fptr);
    sscanf(line, "%f,%f,%f,%f,%i", &batch[0+(i*inputs)], &batch[1+(i*inputs)], &batch[2+(i*inputs)], &batch[3+(i*inputs)], &label);
    (*labels)(i,0) = label;
  }
  float* batchptr = batch;
  layers.emplace_back(batchptr, batch_sz, inputs);
  for (int i = 0; i < hidden; i++) {
    layers.emplace_back(batch_sz, neurons);
  }
  layers.emplace_back(batch_sz, outputs);
  for (int i = 0; i < hidden+1; i++) {
    layers[i].initWeights(layers[i+1]);
  }
}

Eigen::MatrixXd Network::activate(Eigen::MatrixXd matrix)
{
  int nodes = matrix.cols();
  for (int i = 0; i < (matrix.rows()*matrix.cols()); i++) {
    (matrix)((float)i / nodes, i%nodes) = 1.0/(1+exp(-(matrix)((float)i / nodes, i%nodes)));
  }
  return matrix;
}

Eigen::MatrixXd Network::activate_deriv(Eigen::MatrixXd matrix)
{
  int nodes = matrix.cols();
  for (int i = 0; i < (matrix.rows()*matrix.cols()); i++) {
    (matrix)((float)i / nodes, i%nodes) = 1.0/(1+exp(-(matrix)((float)i / nodes, i%nodes))) * (1 - 1.0/(1+exp(-(matrix)((float)i / nodes, i%nodes))));
  }
  return matrix;
}

void Network::feedforward()
{
  for (int i = 0; i < length-1; i++) {
    *layers[i+1].contents = (*layers[i].contents) * (*layers[i].weights);
    for (int j = 0; j < layers[i+1].contents->rows(); j++) {
      // layers[i+1].contents->row(j) += *layers[i+1].bias; TODO ADD ME BACK!
    }
    // if (i != length-2) {
      *layers[i+1].contents = activate(*layers[i+1].contents);
      *layers[i+1].dZ = activate_deriv(*layers[i+1].contents);
    // }
    // else {
    //   *layers[i + 1].dZ = (layers[i + 1].dZ->array() +  1).matrix();
    // }
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
  for (int i = 0; i < layers[length-1].contents->rows(); i++) {
    printf("%lf vs %lf\n", (*labels)(i, 0), (*layers[length-1].contents)(i, 0));
    if ((*labels)(i, 0) == round((*layers[length-1].contents)(i, 0))) {
      correct += 1;
    }
  }
  // std::cout << (1.0/batch_size) * correct << "\n";
  return (1.0/batch_size) * correct;
}

void Network::backpropagate()
{
  // std::cout << "\nROUND\n\n\n\n\n\n";
  std::vector<Eigen::MatrixXd> gradients;
  std::vector<Eigen::MatrixXd> deltas;
  gradients.push_back((((*layers[length-1].contents) - (*labels)).cwiseProduct(((*layers[length-1].contents) - (*labels)))).cwiseProduct(*layers[length-1].dZ));
  deltas.push_back((*layers[length-2].contents).transpose() * gradients[0]);
  int counter = 1;
  for (int i = length-2; i >= 1; i--) {
    // std::cout << gradients[counter-1] << "\n\nTHAT WAS GRADIENT\n\n" <<*layers[i].weights << "\n\nTHAT WAS WEIGHTS\n"
    gradients.push_back((gradients[counter-1] * layers[i].weights->transpose()).cwiseProduct(*layers[i].dZ));
    deltas.push_back((*layers[i-1].contents).transpose() * gradients[counter]);
    // std::cout << gradients[counter] << "\n\nand\n\n" << *layers[i-1].weights << "\n\nweights\n\n" << deltas[counter] << "\n\ndelta above\n\n\n\n\n";
    counter++;
  }
  for (int i = 1; i < gradients.size(); i++) {
    Eigen::MatrixXd gradient = gradients[i];
    *layers[length-2-i].weights -= learning_rate * (deltas[i]);
  }
}

void Network::update_layer(float* vals, int datalen, int index)
{
  for (int i = 0; i < datalen; i++) {
    (*layers[index].contents)((int)i / layers[index].contents->cols(),i%layers[index].contents->cols()) = vals[i];
  }
}

int Network::next_batch(char* path)
{
  FILE* fptr = fopen(path, "r");
  char line[1024] = {' '};
  int inputs = layers[0].contents->cols();
  int datalen = batch_size * inputs;
  float batch[datalen];
  int label = 100;
  for (int i = 0; i < batch_size*batches + 1; i+=batch_size) {
    if (fgets(line, 1024, fptr)==NULL) {
      break;
    }
    if (i >= batches) {
      for (int j = 0; j < batch_size; j++) {
        fgets(line, 1024, fptr);
        // printf("%s", line);
        sscanf(line, "%f,%f,%f,%f,%i", &batch[0 + (j * inputs)],
               &batch[1 + (j * inputs)], &batch[2 + (j * inputs)],
               &batch[3 + (j * inputs)], &label);
        (*labels)(j, 0) = label;
      }
    }
  }
  float* batchptr = batch;
  update_layer(batchptr, datalen, 0);
  fclose(fptr);
  // std::cout << *layers[0].contents << "\n\n and \n\n" << *labels;
  return 0;
}

int prep_file(char* path)
{
  FILE* rptr = fopen(path, "r");
  char line[1024];
  std::vector<std::string> lines;
  int count = 0;
  while (fgets(line, 1024, rptr) != NULL) {
    lines.emplace_back(line);
    count++;
  }
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(lines.begin(), lines.end(), g);
  fclose(rptr);
  std::ofstream out("./shuffled.txt");
  for (int i = 0; i < lines.size(); i++) {
    out << lines[i];
  }
  return count;
}

float Network::test(char* path)
{
  int rounds = 1;
  int exit = 0;
  int linecount = prep_file(path);
  float cost_sum = 0;
  float acc_sum = 0;
  int finalcount;
  for (int i = 0; i < linecount-batch_size; i+=batch_size) {
    feedforward();
    next_batch("./test");
    cost_sum += cost();
    acc_sum += accuracy();
    finalcount = i;
  }
  // std::cout << *layers[0].contents << "\n\n and \n\n" << *labels;
  // std::cout << "TEST COST: " << 1.0/((float) linecount) * totalcost << "\n"
  float chunks = ((float)finalcount/batch_size)+1;
  // std::cout << acc_sum << " " << chunks << " " << acc_sum/chunks << "\n";
  return acc_sum/chunks;
}

void demo(int total_epochs)
{
  auto begin = std::chrono::high_resolution_clock::now();
  // std::cout << "\n\n\n";
  int linecount = prep_file("./data_banknote_authentication.txt");
  Network net ("./shuffled.txt", 4, 2, 1, 5, 10, 1);
  float epoch_cost = 1000;
  float epoch_accuracy = -1;
  int epochs = 0;
  net.batches= 1;
  // net.feedforward();
  // net.backpropagate();
  // std::cout << net.cost() << "\n";

  while (epochs < total_epochs) {
    auto ep_begin = std::chrono::high_resolution_clock::now();
    // int linecount = prep_file("./data_banknote_authentication.txt");
    float cost_sum = 0;
    float acc_sum = 0;
    for (int i = 0; i < linecount-net.batch_size; i+=net.batch_size) {
      net.feedforward();
      net.backpropagate();
      cost_sum += net.cost();
      // std::cout << acc_sum << " "<< net.accuracy() << " " << net.batch_size << "\n";
      acc_sum += net.accuracy();
      // std::cout << net.cost() << " as it is " << net.labels[0] << " vs " << *net.layers[net.length-1].contents << "\n";
      net.batches++;
      int exit = net.next_batch(net.fpath);
      if (exit == -1) {
        break;
      }
    }
    net.batches=1;
    epoch_accuracy = 1.0/((float) linecount/net.batch_size) * acc_sum;
    epoch_cost = 1.0/((float) linecount/net.batch_size) * cost_sum;
    auto ep_end = std::chrono::high_resolution_clock::now();
    printf("Epoch %i/%i - time %f - cost %f - acc %f\n", epochs+1, total_epochs, (double) std::chrono::duration_cast<std::chrono::nanoseconds>(ep_end-ep_begin).count() / pow(10,9), epoch_cost, epoch_accuracy);
    epochs++;
  }
  printf("Test accuracy: %f\n", net.test("./test.txt"));
  // net.list_net();
  auto end = std::chrono::high_resolution_clock::now();
  std::cout <<std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() << " ns aka " << (double) std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / pow(10,9) << "s" << std::endl;
}
