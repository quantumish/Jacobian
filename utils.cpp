#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

// A bunch of hardcoded activation functions. Avoids much of the slowness of custom functions.
// Although the std::function makes it not the fastest way, the functionality is worth it.
// Yes, these functions may be a frustrating to read but they're just equations and I want to conserve space.
double sigmoid(double x) {return 1.0/(1+exp(-x));}
double sigmoid_deriv(double x) {return 1.0/(1+exp(-x)) * (1 - 1.0/(1+exp(x)));}

double linear(double x) {return x;}
double linear_deriv(double x) {return 1;}

double lecun_tanh(double x) {return 1.7159 * tanh((2.0/3) * x);}
double lecun_tanh_deriv(double x) {return 1.14393 * pow(1.0/cosh(2.0/3 * x),2);}

double tanh(double x) {return tanh(x);}
double tanh_deriv(double x) {return pow(1.0/cosh(x),2);}

double inverse_logit(double x) {return (exp(x)/(exp(x)+1));}
double inverse_logit_deriv(double x) {return (exp(x)/pow(exp(x)+1, 2));}

double softplus(double x) {return log(1+exp(x));}
double softplus_deriv(double x) {return exp(x)/(exp(x)+1);}

double cloglog(double x) {return 1-exp(-exp(x));}
double cloglog_deriv(double x) {return exp(x-exp(x));}

double step(double x)
{
  if (x > 0) return 1;
  else return 0;
}
double step_deriv(double x) {return 0;}

double bipolar(double x)
{
  if (x > 0) return 1;
  else return -1;
}
double bipolar_deriv(double x) {return 0;}

std::function<double(double)> rectifier(double (*activation)(double))
{
  auto rectified = [activation](double x) -> double
  { 
    if (x > 0) return (*activation)(x);
    else return 0; 
  };
  return rectified;
}

uintmax_t wc(char const *fname)
{
    static const auto BUFFER_SIZE = 16*1024;
    int fd = open(fname, O_RDONLY);
    if(fd == -1)
      exit(1);

    /* Advise the kernel of our access pattern.  */
    //posix_fadvise(fd, 0, 0, 1);  // FDADVICE_SEQUENTIAL

    char buf[BUFFER_SIZE + 1];
    uintmax_t lines = 0;

    while(size_t bytes_read = read(fd, buf, BUFFER_SIZE))
    {
        if(bytes_read == (size_t)-1)
          exit(1);
        if (!bytes_read)
          break;
        for(char *p = buf; (p = (char*) memchr(p, '\n', (buf + bytes_read) - p)); ++p) {
          printf("START%sEND\n", p);
          ++lines;
        }
    }

    return lines;
}

int istreamtest () {
  std::filebuf fb;
  if (fb.open ("extra.txt",std::ios::in))
  {
    std::istream is(&fb);
    char fchar = '-';
    
    const int MAX_LENGTH = 1024;
    char* line = new char[MAX_LENGTH];
    auto get_begin = std::chrono::high_resolution_clock::now();
    int i = 0;
    auto get_end = std::chrono::high_resolution_clock::now();
    while (is.getline(line, MAX_LENGTH) && strlen(line) > 0 && i < 10) {
      auto get_end = std::chrono::high_resolution_clock::now();
      std::cout << line << "\n";
      i++;
    }
    std::cout << " GET " << std::chrono::duration_cast<std::chrono::nanoseconds>(get_end - get_begin).count() << "\n";
    fb.close();
  }
  return 0;
}
