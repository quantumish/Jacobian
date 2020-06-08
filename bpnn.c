#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* #include "../mapreduce/mapreduce.h" */
/* #include "../mapreduce/server.h" */
/* #include "../mapreduce/worker.h" */

#define BUFSIZE 2048

struct node;

struct edge {
  struct node * source;
  struct node * target;
  double weight;
};

struct node {
  struct edge * incoming;
  struct edge * outgoing;
  float activation;
};

struct layer {
  struct node * nodes;
  int length;
};

struct network {
  struct layer* layers;
  int length;
};

float activate (double value) {
  // Sigmoid
  value = 1/(1+pow(M_E, -value));
}

struct network initialize (char* path, int inputs, int neurons, int layers, int outputs) {
  // Read from CSV
  FILE* fptr = fopen(path, "r");
  double data[inputs];
  fscanf(fptr, "%d,%d,%d,%d,*d", &data[0], &data[1], &data[2], &data[3]);

  // Initialize network
  struct network net;
  net.layers = malloc((layers + 2) * sizeof(struct layer));
  net.length = layers + 2;

  // Initialize input nodes
  net.layers[0].nodes = malloc(inputs * sizeof(struct node));
  for (int i = 0; i < inputs; i++) {
    net.layers[0].nodes[i].activation = activate(data[i]);
  }
  net.layers[0].length = inputs;

  /* // Init hidden layer nodes and output nodes to 0 (will be replaced by feedforward) */
  for (int i = 1; i < layers; i++) {
    net.layers[i].nodes = malloc(neurons * sizeof(struct node));
    for (int j = 0; j < neurons; j++) {
      net.layers[i].nodes[j].activation = 0;
    }
    net.layers[i].length = neurons;
  }
  net.layers[layers-1].nodes = malloc(outputs * sizeof(struct node));
  for (int i = 0; i < outputs; i++) {
    net.layers[layers-1].nodes[i].activation = 0;
  }
  net.layers[layers-1].length = outputs;

  /* // Init edges between layers with random numbers TODO make a function to initialize edges between two layers for the love of God */
  for (int i = 0; i < inputs; i++) {
    net.layers[0].nodes[i].outgoing = malloc(neurons * sizeof(struct edge));
    for (int j = 0; j < neurons; j++) {
      net.layers[1].nodes[j].incoming = malloc(inputs * sizeof(struct edge));
      struct edge connection = {&net.layers[0].nodes[i], &net.layers[1].nodes[j], rand()};
      net.layers[0].nodes[i].outgoing[j] = connection;
      net.layers[1].nodes[j].incoming[i] = connection;
    }
  }
  for (int i = 0; i < layers-1; i++) {
    for (int j = 0; j < neurons; j++) {
      net.layers[i].nodes[j].outgoing = malloc(neurons * sizeof(struct edge));
      for (int k = 0; k < neurons; k++) {
        net.layers[i+1].nodes[k].outgoing = malloc(neurons * sizeof(struct edge));
        struct edge connection = {&net.layers[i].nodes[j], &net.layers[i+1].nodes[k], rand()};
        net.layers[i].nodes[j].outgoing[k] = connection;
        net.layers[i+1].nodes[k].incoming[j] = connection;
      }
    }
  }
  for (int i = 0; i < neurons; i++) {
    net.layers[layers-2].nodes[i].outgoing = malloc(outputs * sizeof(struct edge));
    for (int j = 0; j < outputs; j++) {
      net.layers[layers-1].nodes[j].incoming = malloc(neurons * sizeof(struct edge));
      struct edge connection = {&net.layers[0].nodes[i], &net.layers[1].nodes[j], rand()};
      net.layers[layers-2].nodes[i].outgoing[j] = connection;
      net.layers[layers-1].nodes[j].incoming[i] = connection;
    }
  }
  /* Epic, everything's initialized TODO add biases! */
  return net;
}

void feedforward(struct network net)
{
  for (int i = 1; i < net.length-1; i++) {
    for (int j = 0; j < net.layers[i].length; j++) {
      double sum;
      for (int k = 0; k < net.layers[i-1].length; k++) {
        sum += net.layers[i].nodes[j].incoming->source->activation * net.layers[i].nodes[j].incoming->weight;
      }
      net.layers[i].nodes[j].activation = activate(sum);
      printf("Node %i in layer %i has activation %f", j, i, net.layers[i].nodes[j].activation);
    }
  }
}

/* struct int_pair* map (struct str_pair file) */
/* { */
/*   struct network neuralnet = initialize("./data_banknote_authentication.txt", 4, 5, 2, 2); */
/*   feedforward(neuralnet); */
/* } */

/* struct int_pair* reduce (struct int_pair* input) */
/* { */

/* } */

int main(int argc, char** argv)
{
  struct network neuralnet = initialize("./data_banknote_authentication.txt", 4, 5, 2, 2);
  feedforward(neuralnet);
  /* mapreduce(argv[3], map, reduce, strtol(argv[2], NULL, 10), 50, strtol(argv[1], NULL, 10), 1); */
}
