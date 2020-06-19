#include "bpnn.hpp"

struct pair* map (struct pair input_pair)
{
  char* path = new char[100];
  path = (char*)input_pair.key;
  int linecount = prep_file(path);
  Network* net = new Network (path, 4, 2, 1, 5, 1, 1);  
  auto begin = std::chrono::high_resolution_clock::now();
  // std::cout << "\n\n\n";
  float epoch_cost = 1000;
  float epoch_accuracy = -1;
  int epochs = 0;
  net->batches= 0;
  // net.feedforward();
  // net.backpropagate();
  // std::cout << net.cost() << "\n";

  printf("Beginning train on %i instances for %i epochs...\n", linecount, 50);
  while (epochs < 50) {
    auto ep_begin = std::chrono::high_resolution_clock::now();
    // int linecount = prep_file("./data_banknote_authentication.txt");
    float cost_sum = 0;
    float acc_sum = 0;
    double times[5] = {0};
    for (int i = 0; i <= linecount-net->batch_size; i+=net->batch_size) {
      auto feed_begin = std::chrono::high_resolution_clock::now();
      net->feedforward();
      auto back_begin = std::chrono::high_resolution_clock::now();
      net->backpropagate();
      auto cost_begin = std::chrono::high_resolution_clock::now();
      cost_sum += net->cost();
      // std::cout << acc_sum << " "<< net.accuracy() << " " << net.batch_size << "\n";
      auto acc_begin = std::chrono::high_resolution_clock::now();
      acc_sum += net->accuracy();
      // std::cout << net.cost() << " as it is " << net.labels[0] << " vs " << *net.layers[net.length-1].contents << "\n";
      auto batch_begin = std::chrono::high_resolution_clock::now();

      int exit = net->next_batch(net->fpath);
      auto loop_end = std::chrono::high_resolution_clock::now();
      times[0] += std::chrono::duration_cast<std::chrono::nanoseconds>(back_begin - feed_begin).count() / pow(10,9);
      times[1] += std::chrono::duration_cast<std::chrono::nanoseconds>(cost_begin - back_begin).count() / pow(10,9);
      times[2] += std::chrono::duration_cast<std::chrono::nanoseconds>(acc_begin - cost_begin).count() / pow(10,9);
      times[3] += std::chrono::duration_cast<std::chrono::nanoseconds>(batch_begin - acc_begin).count() / pow(10,9);
      times[4] += std::chrono::duration_cast<std::chrono::nanoseconds>(loop_end - batch_begin).count() / pow(10,9);
      net->batches++;
      if (exit == -1) {
        break;
      }
    }
    printf("Avg time spent across %i batches: %lf on feedforward, %lf on backprop, %lf on cost, %lf on acc, %lf on next batch\n", net->batches, times[0]/net->batches, times[1]/net->batches, times[2]/net->batches, times[3]/net->batches, times[4]/net->batches);
    net->batches=1;
    epoch_accuracy = 1.0/((float) linecount/net->batch_size) * acc_sum;
    epoch_cost = 1.0/((float) linecount/net->batch_size) * cost_sum;
    auto ep_end = std::chrono::high_resolution_clock::now();
    printf("Epoch %i/%i - time %f - cost %f - acc %f\n", epochs+1, 50, (double) std::chrono::duration_cast<std::chrono::nanoseconds>(ep_end-ep_begin).count() / pow(10,9), epoch_cost, epoch_accuracy);
    epochs++;
  }
  struct pair* output = new struct pair[2];
  output[0].key = path;
  output[0].value = net;
  
  output[1].key = 0x0;
  output[1].value = 0x0;
  return output;
}

struct pair* reduce (struct pair* input_pairs)
{
  printf("%p %p VALS\n", input_pairs[0].key, input_pairs[0].value);
  struct pair* output = new struct pair[6];  
  for (int i = 0; i < 2; i++) {
    // Network net = *(Network*)input_pairs[i].value;
    // float* cost = new float;
    // *cost = net.test("./test.txt");
    // output[i].key = input_pairs[i].key;
    // output[i].value = cost;
  }
  output[5].key = 0x0;
  output[5].value = 0x0;
  return output;
}

void translate(char* path)
{
  FILE* rptr = fopen(path, "r");
  FILE* wptr = fopen("./translated", "w");
  char* line = new char[MAXLINE];
  char* newline = new char[MAXLINE];
  while (fgets(line, MAXLINE, rptr) != NULL) {
    void* addr1;
    void* addr2;
    sscanf(line, "%p %p", &addr1, &addr2);
    sprintf(newline, "%s %f", (char*)addr1, *(float*)addr2);
    int batch_num = strtol((char*)addr1, NULL, 10);
    fprintf(wptr, "%s %f",(char*)addr1, *(float*)addr2);
  }
  fclose(rptr);
  fclose(wptr);
  free(newline);
  free(line);
}

int main(int argc, char** argv)
{
  begin(argv[2], map, reduce, translate, strtol(argv[1], NULL, 10), 2, argv[3], strtol(argv[4], NULL, 10));
}
