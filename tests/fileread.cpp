//
//  fileread.cpp
//  Jacobian
//
//  Created by David Freifeld
//  Copyright © 2020 David Freifeld. All rights reserved.
//

uintmax_t wc(char const *fname)
{
    static const auto BUFFER_SIZE = 1024;
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
          printf("About to segfault? It's simple - just don't!\n");
          //          printf("%s", p)
          for (int i = 1; i < BUFFER_SIZE; i++) {
            printf("%c", p[i]);
          }
          printf("\nGlad you heeded my advice.\n");
          printf("\n\nDONE\n\n");
          ++lines;
          //printf(" DONE \n");
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
