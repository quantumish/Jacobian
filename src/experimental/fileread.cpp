//
//  fileread.cpp
//  Jacobian
//
//  Created by David Freifeld
//
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <ctime>
#include <chrono>
#include <iostream>
#include <cmath>
#include <stdint.h>

// No safety checks whatsoever. Use at your own risk.
// Assumes solely numeric input. No NaN, no inf.
// float strtof_but_bad(char* p)
// {
//     float flt;
//     for (int i = 0; 
// }

int main () {
    auto start = std::chrono::high_resolution_clock::now();
    static const auto BUFFER_SIZE = 16*1024;
    int fd = open("../../data_banknote_authentication.txt", O_RDONLY & O_NONBLOCK);
    if(fd == -1) {
        printf("fd == -1\n");
        return 1;
    }

    /* Advise the kernel of our access pattern.  */
    // posix_fadvise(fd, 0, 0, 1);  // FDADVICE_SEQUENTIAL
    
    char buf[BUFFER_SIZE + 1];
    uintmax_t lines = 0;
    float tmp;
    int bufs = 0;
    while(size_t bytes_read = read(fd, buf, BUFFER_SIZE))
    {
        if(bytes_read == (size_t)-1) {
            printf("bytes_read == (size_t)-1\n");
            return 1;
        }
        if (!bytes_read) break;
        for(char *p = buf;; ++p) {
            char* bound = (char*) memchr(p+1, '\n', (buf + bytes_read) - p+1);
            if (bound - p < 0) break; // Stop.
            tmp = strtod(p, NULL); // Read first float
            for (; p < bound; ++p) { 
                if (*p ==  ',') {
                    tmp = strtod(p, NULL); // Read float following delimiter
                }
            }
            ++lines;
        }
        bufs++;
    }
    printf("Final %f, %i bufs\n", tmp, bufs);
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / pow(10,9);
    
    auto f_start = std::chrono::high_resolution_clock::now();
    int flines = 0;
    FILE* fptr = fopen("../../data_banknote_authentication.txt", "r");
    char fbuf[1024];
    float ftmp;
    while (fgets(fbuf, 1024, fptr) != NULL) {
        char *p;
        p = strtok(fbuf,",");
        for (int j = 0; j < 4; j++) {
            ftmp = strtod(p, NULL);
            p = strtok(NULL,",");
        }
        ftmp = strtod(p, NULL);
        flines++;
    }
    auto f_end = std::chrono::high_resolution_clock::now();
    if (flines == lines) std::cout << "Linecount is valid!" << "\n";
    else std::cout << "WARN: INVALID LINECOUNT " << lines << " vs. " << flines << "\n";
    if (ftmp == tmp) std::cout << "Final value read is valid!" << "\n";
    else std::cout << "WARN: INVALID LAST READ VAL " << tmp << " vs. " << ftmp << "\n";
    double ftime = std::chrono::duration_cast<std::chrono::nanoseconds>(f_end - f_start).count() / pow(10,9);
    std::cout << ftime << " " << time << " so " << ftime/time << "% speedup\n";
}
