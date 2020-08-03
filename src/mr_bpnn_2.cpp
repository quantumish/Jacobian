//
//  mr_bpnn_2.cpp
//  Jacobian
//
//  Created by David Freifeld
//

#include "bpnn.hpp"
struct pair* map (struct pair input_pair)
{
  char* path = new char[100];
  strcpy(path, (char*)input_pair.key);
  strcat(path, "_shuf");
  printf("%s and %s\n", path, (char*)input_pair.key);
  int linecount = prep_file((char*)input_pair.key, path);
  Network* net = new Network(path, 16, 0.0155, 0.03, 2, 0, 0.9);
  for (int i = 0; i < 50; i++) {
    net->train();
  }
  struct pair* output = new struct pair;
  char* key = new char[100];
  strcpy(key, path);
  output[0].key = key;
  output[0].value = net;
  return output;
}

struct pair* reduce (struct pair* input_pairs)
{
  struct pair* output = new struct pair[6];  
  for (int i = 0; input_pairs[i].key != 0x0; i++) {
    float* cost = new float;
    *cost = ((Network*)input_pairs[i].value)->get_val_cost();
    output[i].key = input_pairs[i].key;
    output[i].value = cost;
  }
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
    fprintf(wptr, "%s %f\n",(char*)addr1, *(float*)addr2);
  }
  fclose(rptr);
  fclose(wptr);
  free(newline);
  free(line);
}

int main()
{
  begin("./data_banknote_authentication.txt", map, reduce, translate, 1, 2, "108.169.4.115", 1);
}
