#include "bpnn.hpp"

struct pair* map (struct pair input_pair)
{
  pair* output_pairs = new pair[1024];
  int linecount = prep_file("./data_banknote_authentication.txt");
  Network net ("./shuffled.txt", 4, 2, 1, 5, 1, 1);
  float epoch_cost = 1000;
  int epochs = 0;
  net.batches= 1;

  while (epochs < 1) {
    int linecount = prep_file("./data_banknote_authentication.txt");
    float cost_sum = 0;
    for (int i = 0; i < linecount-net.batch_size; i+=net.batch_size) {
      net.feedforward();
      net.backpropagate();
      cost_sum += net.cost();
      net.batches++;
      int exit = net.next_batch();
      if (exit == -1) {
        break;
      }
    }
    net.batches=1;
    epoch_cost = 1.0/((float) linecount) * cost_sum;
    printf("EPOCH %i: Cost is %f for %i instances.\n", epochs, epoch_cost, linecount);
    epochs++;
  }
  int rounds = 1;
  int exit = 0;
  float totalcost = -1;
  while (exit == 0) {
    FILE* fptr = fopen((char*)input_pair.key, "r");
    char line[1024] = {' '};
    int inputs = net.layers[0].contents->cols();
    int datalen = net.batch_size * inputs;
    float batch[datalen];
    for (int i = 0; i < net.batch_size*rounds + 1; i++) {
      if (fgets(line, 1024, fptr)==NULL) {
        exit = -1;
      }
      if (i >= rounds) {
        for (int j = 0; j < net.batch_size; j++) {
          fgets(line, 1024, fptr);
          sscanf(line, "%f,%f,%f,%f,%lf", &batch[0 + (j * inputs)], &batch[1 + (j * inputs)], &batch[2 + (j * inputs)], &batch[3 + (j * inputs)], &(*net.labels)(j));
        }
      }
    }
    float *batchptr = batch;
    net.update_layer(batchptr, datalen, 0);
    net.feedforward();
    for (int i = 0; i < net.batch_size; i++) {
      char* key = new char[1024];
      sprintf(key, "%i", net.batch_size);
      output_pairs[i].key = (void*) key;
      float cost = net.cost();
      output_pairs[i].value = (void*) &cost;
    }
    totalcost += net.cost();
    rounds++;
  }
  return output_pairs;
}

pair* reduce (pair* input_pairs)
{
  pair* output_pairs = new pair[2];
  int keysum = 0;
  int valsum = 0;
  for (int i = 0; i < 5; i++) {
    printf("%s %i\n", (char*)input_pairs[i].key, *(int*)input_pairs[i].value);
    keysum += strtol((char*)input_pairs[i].key, NULL, 10);
    valsum += *(int*)input_pairs[i].value;
  }
  char key[1024];
  std::cout << keysum << " and " << valsum << " are SUMS\n";
  sprintf(key, "%d", keysum);
  output_pairs[0].key = (void*) key;
  output_pairs[0].value = &valsum;
  output_pairs[1].key = (void*) '\0';
  int nullval = -1;
  output_pairs[1].value = &nullval;
  return output_pairs;
}

int main(int argc, char** argv)
{
  begin(argv[2], map, reduce, strtol(argv[1], NULL, 10), 1, argv[3], strtol(argv[4], NULL, 10));
}
