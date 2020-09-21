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
    int fd = open("../../data_banknote_authentication.txt", O_RDONLY);
    if(fd == -1) {
        printf("fd == -1\n");
        return 1;
    }

    /* Advise the kernel of our access pattern.  */
    // posix_fadvise(fd, 0, 0, 1);  // FDADVICE_SEQUENTIAL
    
    char buf[BUFFER_SIZE + 1];
    uintmax_t lines = 0;
    float tmp;
    while(size_t bytes_read = read(fd, buf, BUFFER_SIZE))
    {
        if(bytes_read == (size_t)-1) {
            printf("bytes_read == (size_t)-1\n");
            return 1;
        }
        if (!bytes_read) break;
        for(char *p = buf; (p = (char*) memchr(p, '\n', (buf + bytes_read) - p)); ++p) {
            long bound = (char*) memchr(p+1, '\n', (buf + bytes_read) - p+1) - p;
            if (bound < 0) break; // Stop.
            //            printf("%p (%p + %ld) vs %p\n", p+bound, p, bound, buf+BUFFER_SIZE);
            tmp = strtod(p, NULL); // Read first float
            //int delims = 0;
            //            printf("%f\n", tmp);
            for (int i = 0; i < bound; i++) { 
                if (*(p+i) ==  ',') {
                    tmp = strtod(p+i+1, NULL); // Read float following delimiter
                    //printf("%f\n", tmp);
                    // delims++;
                    // printf("Comma: %f %d\n", tmp, delims);
                }
            }
            ++lines;
        }
        
    }
    printf("Final %f\n", tmp);
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
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
    if (ftmp == tmp) std::cout << "Final value read is valid!" << "\n";
    double ftime = std::chrono::duration_cast<std::chrono::nanoseconds>(f_end - f_start).count();
    std::cout << ftime << " " << time << " " << ftime/time << "\n";
}
