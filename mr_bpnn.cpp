#include "bpnn.hpp"
#include "utils.hpp"

class NetworkArray
{
public:
  std::function<struct pair*(struct pair*)> map;
  std::function<struct pair*(struct pair*)> reduce;
  std::function<void(char*)> translate;
  
  NetworkArray(char* configuration, std::function<Network*(void)> setup, int epochs);
  void start_array(char* data, int m, int length, char* ip, int r);
};

NetworkArray::NetworkArray(char* configuration, std::function<Network*(void)> setup, int epochs)
{
  map = [setup, epochs](struct pair* input_pair) -> struct pair*
  {
    printf("Recieved %p %p in form of %p\n", input_pair->key, input_pair->value, input_pair);
    printf("%s %p\n", (char*)input_pair->key, input_pair->value);
    char* path = new char[100];
    strcpy(path, (char*)input_pair->key);
    //strcat(path, "_shuf");
    //int linecount = prep_file((char*)input_pair.key, path);
    // Network* net = setup();
    // net->train(epochs);
    struct pair* output = new struct pair;
    //char* key = new char[100];
    //strcpy(key, path);
    output[0].key = 0x0;
    output[0].value = 0x0;
    return output;
  };
  reduce = [](struct pair* input_pairs) -> struct pair*
  {
    struct pair* output = new struct pair[6];  
    for (int i = 0; input_pairs[i].key != 0x0; i++) {
      float* acc = new float;
      *acc = ((Network*)input_pairs[i].value)->test("./test.txt");
      output[i].key = input_pairs[i].key;
      output[i].value = acc;
    }
    return output;
  };
  translate = [](char* path) -> void
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
  };
}

void NetworkArray::start_array(char* data, int m, int length, char* ip, int r)
{
  begin(data, map, reduce, translate, m, length, ip, r);
}

Network* setup()
{
  Network* net = new Network ("./data_banknote_authentication.txt", 10, 0.01, 0.001, 0.9);
  net->add_layer(4, "linear");
  net->add_layer(5, "sigmoid");
  net->set_activation(1, lecun_tanh, lecun_tanh_deriv);
  net->add_layer(1, "resig");
  net->initialize();
  return net; 
}

int main()
{
  NetworkArray netarray ("Train", setup, 1);
  // char* path = "./extra.txt";
  // char* msg = "junk";
  //struct pair testing = {(void*)path, (void*)msg};
  //  printf("SENDING %p %p (part of %p)\n", testing.key, testing.value, &testing);
  //netarray.map(testing);
  netarray.start_array("./extra.txt", 5, 1, "98.33.105.140", 1);
  return 0;
}

