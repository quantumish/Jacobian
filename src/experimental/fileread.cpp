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

int main (){
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

    while(size_t bytes_read = read(fd, buf, BUFFER_SIZE))
    {
        if(bytes_read == (size_t)-1) {
            printf("bytes_read == (size_t)-1\n");
            return 1;
        }
        if (!bytes_read) break;
        for(char *p = buf; (p = (char*) memchr(p, '\n', (buf + bytes_read) - p)); ++p) {
            ++lines;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    auto f_start = std::chrono::high_resolution_clock::now();
    int flines = 0;
    FILE* fptr = fopen("../../data_banknote_authentication.txt", "r");
    char fbuf[1024];
    while (fgets(fbuf, 1024, fptr) != NULL) flines++;
    auto f_end = std::chrono::high_resolution_clock::now();
    std::cout << flines << " " << lines << "\n";
    double ftime = std::chrono::duration_cast<std::chrono::nanoseconds>(f_end - f_start).count();
    std::cout << ftime << " " << time << " " << ftime/time << "\n";
}
