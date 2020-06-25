#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

double sigmoid(double x)
{
  return 1.0/(1+exp(-x));
}

double sigmoid_deriv(double x)
{
  return 1.0/(1+exp(-x)) * (1 - 1.0/(1+exp(x)));
}

double resig(double x)
{
  if (x > 0) return 1.0/(1+exp(-x));
  else return 0;
}

double resig_deriv(double x)
{
  if (x > 0) return 1.0/(1+exp(-x)) * (1 - 1.0/(1+exp(x)));
  else return 0;
}

double linear(double x)
{
  return x;
}

double linear_deriv(double x)
{
  return 1;
}

double relu(double x)
{
  if (x > 0) return x;
  else return 0;
}

double relu_deriv(double x)
{
  if (x > 0) return 1;
  else return 0;
}

static uintmax_t wc(char const *fname)
{
    static const auto BUFFER_SIZE = 16*1024;
    int fd = open(fname, O_RDONLY);
    if(fd == -1)
      exit(1);

    /* Advise the kernel of our access pattern.  */
    //    posix_fadvise(fd, 0, 0, 1);  // FDADVICE_SEQUENTIAL

    char buf[BUFFER_SIZE + 1];
    uintmax_t lines = 0;

    while(size_t bytes_read = read(fd, buf, BUFFER_SIZE))
    {
        if(bytes_read == (size_t)-1)
          exit(1);
        if (!bytes_read)
          break;

        for(char *p = buf; (p = (char*) memchr(p, '\n', (buf + bytes_read) - p)); ++p)
          printf("Line? %s END\n", buf);
          ++lines;
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
