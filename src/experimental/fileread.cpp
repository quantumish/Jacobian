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

// Courtesy of github.com/exr0n
typedef float val_t;
inline float scan(char **p)
{
    float n;
    int neg = 1;
    while (!isdigit(**p) && **p != '-' && **p != '.') ++*p;
    if (**p == '-') neg = -1, ++*p;
    for (n=0; isdigit(**p); ++*p) (n *= 10) += (**p-'0');
    if (*(*p)++ != '.') return n*neg;
    float d = 1;
    for (; isdigit(**p); ++*p) n += (d /= 10) * (**p-'0');
    return n*neg;
}

float current()
{
    static const auto BUFFER_SIZE = 16*1024;
    int fd = open("../../data_banknote_authentication.txt", O_RDONLY & O_NONBLOCK);
    if(fd == -1) {
        printf("fd == -1\n");
        return 1;
    }
    
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
        for(char *p = buf;;) {
            char* bound = (char*) memchr(p, '\n', (buf + bytes_read) - p);
            if (bound - p < 0) break; // Stop.
            for (int i=0; i<5; ++i) tmp = scan(&p);
            p = bound + 1;
            ++lines;
        }
    }
    return tmp;
}

void prep()
{
    FILE* wptr = fopen("../../data_banknote_authentication.bin", "wb");
    static const auto BUFFER_SIZE = 512*1024;
    int fd = open("../../data_banknote_authentication.txt", O_RDONLY | O_NONBLOCK);
    if(fd == -1) {
        printf("fd == -1\n");
    }
    float tmp;
    char buf[BUFFER_SIZE + 1];
    while(size_t bytes_read = read(fd, buf, BUFFER_SIZE))
    {
        if(bytes_read == (size_t)-1) {
            printf("bytes_read == (size_t)-1\n");
        }
        if (!bytes_read) break;
        for(char *p = buf;;) {
            char* bound = (char*) memchr(p, '\n', (buf + bytes_read) - p);
            if (bound - p < 0) break; // Stop.
            for (int i=0; i<5; ++i) {
                tmp = scan(&p);
                fwrite((void*)&tmp, sizeof(float), 1, wptr);
            }
            p = bound + 1;
        }
    }
}

float newer()
{
    static const auto BUFFER_SIZE = 600*1024;
    int fd = open("../../data_banknote_authentication.bin", O_RDONLY | O_NONBLOCK);
    uintmax_t maxlines = 1003222;
    if(fd == -1) {
        printf("Cannot open file.");
        exit(1);
    }
    char buf[BUFFER_SIZE];
    uintmax_t lines = 0;
    float tmp;
    while(size_t bytes_read = read(fd, buf, BUFFER_SIZE))
    {
        if(bytes_read == (size_t)-1) {
            printf("bytes_read == (size_t)-1\n");
            exit(1);
        }
        if (!bytes_read) break;
        // for(char *p = buf; p < buf+BUFFER_SIZE;) {
        //     for (int i=0; i<5; ++i) {
        //         tmp = *((float*)p);
        //         p += sizeof(float);
        //     }
        //     ++lines;
        // }
    }
    return tmp;
}

float newest()
{
    static const auto BUFFER_SIZE = 600*1024;
    int fd = open("../../data_banknote_authentication.bin", O_RDONLY | O_NONBLOCK);
    fcntl(fd, F_RDADVISE);
    uintmax_t maxlines = 1003222;
    if(fd == -1) {
        printf("Cannot open file.");
        exit(1);
    }
    char buf[BUFFER_SIZE];
    uintmax_t lines = 0;
    float tmp;
    while(size_t bytes_read = read(fd, buf, BUFFER_SIZE))
    {
        if(bytes_read == (size_t)-1) {
            printf("bytes_read == (size_t)-1\n");
            exit(1);
        }
        if (!bytes_read) break;
        // for(char *p = buf; p < buf+BUFFER_SIZE;) {
        //     for (int i=0; i<5; ++i) {
        //         tmp = *((float*)p);
        //         p += sizeof(float);
        //     }
        //     ++lines;
        // }
    }
    return tmp;
}

int main () {
    int epochs = 1;
    float tmp;
    auto start = std::chrono::high_resolution_clock::now();
    //prep();
    auto prep_end = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < epochs; i++) {
        tmp = newest();
    }
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - prep_end + prep_end-start).count() / pow(10,9);
    double runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - prep_end).count() / pow(10,9);
    float ftmp;
    auto f_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < epochs; i++) {
        ftmp = newer();
    }
    auto f_end = std::chrono::high_resolution_clock::now();
    // if (flines == lines) std::cout << "Linecount is valid!" << "\n";
    // else std::cout << "WARN: INVALID LINECOUNT " << lines << " vs. " << flines << "\n";
    if (ftmp == tmp) std::cout << "Final value read is valid!" << "\n";
    else std::cout << "WARN: INVALID LAST READ VAL " << tmp << " vs. " << ftmp << "\n";
    double ftime = std::chrono::duration_cast<std::chrono::nanoseconds>(f_end - f_start).count() / pow(10,9);
    std::cout << ftime << " " << time << " " << runtime << " (" << ftime/time * 100 << "% speedup overall, " << ftime/runtime * 100 << "% speedup runtime)\n";
}
