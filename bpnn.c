#include <stdio.h>

#include "../mapreduce/mapreduce.h"
#include "../mapreduce/server.h"
#include "../mapreduce/worker.h"

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

float activate (float value) {

}

struct network initialize (char* path, int inputs, int neurons, int layers, int outputs) {
  struct layer input;
  for (int i = 0; i < 4; i++) {
    input.nodes[i].activation = 0; // Needs some sort of scaling
  }
}

struct int_pair* map (struct str_pair file)
{
  
}

struct int_pair* reduce (struct int_pair* input)
{

}

int main(int argc, char** argv)
{
  mapreduce(argv[3], map, reduce, strtol(argv[2], NULL, 10), 50, strtol(argv[1], NULL, 10), 1);
}
