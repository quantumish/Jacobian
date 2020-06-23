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

// Testing 123
Network::Network(char* path, int inputs, int hidden, int outputs, int neurons, int batch_sz, float rate)
{
  learning_rate = rate;
  instances = prep_file(path, "./shuffled.txt");
  length = hidden + 2;
  batch_size = batch_sz;
  data = fopen("./shuffled.txt", "r");
  int datalen = batch_sz*inputs;
  float batch[datalen];
  labels = new Eigen::MatrixXd (batch_size, 1);
  int label;
  char line[1024] = {' '};
  for (int i = 0; i < batch_size; i++) {
    fgets(line, 1024, data);
    sscanf(line, "%f,%f,%f,%f,%i", &batch[0+(i*inputs)], &batch[1+(i*inputs)], &batch[2+(i*inputs)], &batch[3+(i*inputs)], &label);
    (*labels)(i,0) = label;
  }
  float* batchptr = batch;
  layers.emplace_back(batchptr, batch_size, inputs);
  for (int i = 0; i < hidden; i++) {
    layers.emplace_back(batch_size, neurons);
  }
  layers.emplace_back(batch_size, outputs);
  for (int i = 0; i < hidden+1; i++) {
    layers[i].initWeights(layers[i+1]);
  }
  batches = 1;
}

Eigen::MatrixXd Network::activate(Eigen::MatrixXd matrix)
{
  int nodes = matrix.cols();
  for (int i = 0; i < (matrix.rows()*matrix.cols()); i++) {
    if ((matrix)((float)i / nodes, i%nodes) > 0) {
      (matrix)((float)i / nodes, i%nodes) = 1.0/(1+exp(-(matrix)((float)i / nodes, i%nodes)));
    }
    else {
      (matrix)((float)i / nodes, i%nodes) = 0;
    }
  }
  return matrix;
}

Eigen::MatrixXd Network::activate_deriv(Eigen::MatrixXd matrix)
{
  int nodes = matrix.cols();
  for (int i = 0; i < (matrix.rows()*matrix.cols()); i++) {
    if ((matrix)((float)i / nodes, i%nodes) > 0) {
      (matrix)((float)i / nodes, i%nodes) = 1.0/(1+exp(-(matrix)((float)i / nodes, i%nodes))) * (1 - 1.0/(1+exp(-(matrix)((float)i / nodes, i%nodes))));
    }
    else {
      (matrix)((float)i / nodes, i%nodes) = 0;
    }
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
    *layers[i+1].contents = activate(*layers[i+1].contents);
    *layers[i+1].dZ = activate_deriv(*layers[i+1].contents);
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
    deltas.push_back((*layers[i-1].contents).transpose() * gradients[counter]);
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

int Network::next_batch()
{
  char line[1024] = {' '};
  int inputs = layers[0].contents->cols();
  int datalen = batch_size * inputs;
  float batch[datalen];
  int label = -1;

  for (int i = 0; i < batch_size; i++) {
    if (fgets(line, 1024, data)==NULL) {
      break;
    }
    sscanf(line, "%f,%f,%f,%f,%i", &batch[0 + (i * inputs)],
           &batch[1 + (i * inputs)], &batch[2 + (i * inputs)],
           &batch[3 + (i * inputs)], &label);
    (*labels)(i, 0) = label;
  }
  float* batchptr = batch;
  update_layer(batchptr, datalen, 0);
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
  while (epochs < total_epochs) {
    auto ep_begin = std::chrono::high_resolution_clock::now();
    float cost_sum = 0;
    float acc_sum = 0;
    for (int i = 0; i <= instances-batch_size; i+=batch_size) {
      feedforward();
      backpropagate();
      cost_sum += cost();
      acc_sum += accuracy();
      if (i != instances-batch_size) { // Don't try to advance batch on final batch.
        next_batch();
      }      batches++;
    }
    epoch_accuracy = 1.0/((float) instances/batch_size) * acc_sum;
    epoch_cost = 1.0/((float) instances/batch_size) * cost_sum;
    auto ep_end = std::chrono::high_resolution_clock::now();
    double epochtime = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(ep_end-ep_begin).count() / pow(10,9);
    printf("Epoch %i/%i - time %f - cost %f - acc %f\n", epochs+1, total_epochs, epochtime, epoch_cost, epoch_accuracy);
    batches=1;
    epochs++;
  }
}

void demo(int total_epochs)
{
  int linecount = prep_file("./extra.txt", "./shuffled.txt");
  Network net ("./shuffled.txt", 4, 1, 1, 5, 10, 1);
  float epoch_cost = 1000;
  float epoch_accuracy = -1;
  int epochs = 0;

  printf("Beginning train on %i instances for %i epochs...\n", linecount, total_epochs);
  while (epochs < total_epochs) {
    auto ep_begin = std::chrono::high_resolution_clock::now();
    float cost_sum = 0;
    float acc_sum = 0;
    double times[5] = {0};
    for (int i = 0; i <= linecount-net.batch_size; i+=net.batch_size) {
      auto feed_begin = std::chrono::high_resolution_clock::now();
      net.feedforward();
      auto back_begin = std::chrono::high_resolution_clock::now();
      net.backpropagate();
      auto cost_begin = std::chrono::high_resolution_clock::now();
      cost_sum += net.cost();
      auto acc_begin = std::chrono::high_resolution_clock::now();
      acc_sum += net.accuracy();
      auto batch_begin = std::chrono::high_resolution_clock::now();

      if (i != linecount-net.batch_size) { // Don't try to advance batch on final batch.
        net.next_batch();
      }
      auto loop_end = std::chrono::high_resolution_clock::now();
      times[0] += std::chrono::duration_cast<std::chrono::nanoseconds>(back_begin - feed_begin).count() / pow(10,9);
      times[1] += std::chrono::duration_cast<std::chrono::nanoseconds>(cost_begin - back_begin).count() / pow(10,9);
      times[2] += std::chrono::duration_cast<std::chrono::nanoseconds>(acc_begin - cost_begin).count() / pow(10,9);
      times[3] += std::chrono::duration_cast<std::chrono::nanoseconds>(batch_begin - acc_begin).count() / pow(10,9);
      times[4] += std::chrono::duration_cast<std::chrono::nanoseconds>(loop_end - batch_begin).count() / pow(10,9);
      net.batches++;
    }
    epoch_accuracy = 1.0/((float) linecount/net.batch_size) * acc_sum;
    epoch_cost = 1.0/((float) linecount/net.batch_size) * cost_sum;
    auto ep_end = std::chrono::high_resolution_clock::now();
    double epochtime = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(ep_end-ep_begin).count() / pow(10,9);
    printf("Epoch %i/%i - time %f - cost %f - acc %f\n", epochs+1, total_epochs, epochtime, epoch_cost, epoch_accuracy);
    printf("Avg time spent across %i batches: %lf on feedforward, %lf on backprop, %lf on cost, %lf on acc, %lf on next batch.\n", net.batches, times[0]/net.batches, times[1]/net.batches, times[2]/net.batches, times[3]/net.batches, times[4]/net.batches);
    printf("Time spent across epoch: %lf on feedforward, %lf on backprop, %lf on cost, %lf on acc, %lf on next batch, %lf other.\n", times[0], times[1], times[2], times[3], times[4], epochtime-times[0]-times[1]-times[2]-times[3]-times[4]);

    net.batches=1;
    epochs++;
  }  
  // float newvals[4] = {0};
  // FILE* new = fopen("./predict.txt", "r");
  // fscanf(new, "%f, %f, %f, %f", &newvals[0], &newvals[1], &newvals[2], &newvals[3]);
  // net.update_layer(newvals, 4, 0);
  // net.feedforward();
  // net.list_net();
  //net.list_net();
  //  printf("Test accuracy: %f\n", net.test("./test.txt"));
}
