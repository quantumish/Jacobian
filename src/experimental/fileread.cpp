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
#include <sys/mman.h>
#include <sys/uio.h>

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

float mmap_read()
{
    static const auto BUFFER_SIZE = 600*1024;
    int fd = open("../../data_banknote_authentication.big.bin", O_RDONLY | O_NONBLOCK);
    if(fd == -1) {
        printf("Cannot open file.");
        exit(1);
    }
    int rc, ii;
    struct stat st;
    size_t size;
    rc = fstat(fd, &st);
    size=st.st_size;
    float* ptr = (float*) mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
    madvise(ptr, size, POSIX_MADV_SEQUENTIAL);
    char buf[BUFFER_SIZE];
    uintmax_t lines = 0;
    float tmp;
    for (ii=0; ii < size/sizeof *ptr; ii++) {
        // Nothin.
    }
    rc = munmap(ptr, size);
    close(fd);
    return tmp;
}

float vec_read()
{
#define CHUNK_SZ (200*1024)
#define BUFFER_SZ (600*1024)
#define NUM_CHUNKS (BUFFER_SZ/CHUNK_SZ)
    int fd = open("../../data_banknote_authentication.big.bin", O_RDONLY | O_NONBLOCK);
    char rawbuf[BUFFER_SZ];
    char* buf = (char*) rawbuf;
    iovec iovecs[NUM_CHUNKS];
    for (int i = 0; i < BUFFER_SZ; i+=CHUNK_SZ) {
        iovecs[i/CHUNK_SZ].iov_base = buf + i;
        iovecs[i/CHUNK_SZ].iov_len = CHUNK_SZ;
    }
    while(size_t bytes_read = readv(fd, iovecs, NUM_CHUNKS))
    {
        if(bytes_read == (size_t)-1) {
            printf("\n%zu\n", bytes_read);
            printf("bytes_read == (size_t)-1\n");
            exit(1);
        }
        if (!bytes_read) break;
    }
    return 0;
}

float std_read()
{
    int BUFFER_SIZE = 512*1024;
    int fd = open("../../data_banknote_authentication.big.bin", O_RDONLY | O_NONBLOCK);
    fcntl(fd, F_RDADVISE);
    char buf[BUFFER_SIZE];
    float tmp;
    while(size_t bytes_read = read(fd, buf, BUFFER_SIZE))
    {
        if(bytes_read == (size_t)-1) {
            printf("bytes_read == (size_t)-1\n");
            exit(1);
        }
        if (!bytes_read) break;
    }
    return tmp;
}

float byte_read()
{
    int BUFFER_SIZE = 512*1024;
    int fd = open("../../data_banknote_authentication.big.bin", O_RDONLY | O_NONBLOCK);
    fcntl(fd, F_RDADVISE);
    std::byte buf[BUFFER_SIZE];
    float tmp;
    while(size_t bytes_read = read(fd, buf, BUFFER_SIZE))
    {
        if(bytes_read == (size_t)-1) {
            printf("bytes_read == (size_t)-1\n");
            exit(1);
        }
        if (!bytes_read) break;
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
        tmp = std_read();
    }
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - prep_end + prep_end-start).count() / pow(10,9);
    double runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - prep_end).count() / pow(10,9);
    float ftmp;
    auto f_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < epochs; i++) {
        ftmp = byte_read();
    }
    auto f_end = std::chrono::high_resolution_clock::now();
    // if (flines == lines) std::cout << "Linecount is valid!" << "\n";
    // else std::cout << "WARN: INVALID LINECOUNT " << lines << " vs. " << flines << "\n";
    if (ftmp == tmp) std::cout << "Final value read is valid!" << "\n";
    else std::cout << "WARN: INVALID LAST READ VAL " << tmp << " vs. " << ftmp << "\n";
    double ftime = std::chrono::duration_cast<std::chrono::nanoseconds>(f_end - f_start).count() / pow(10,9);
    std::cout << ftime << " " << time << " so " << 9/ftime << " GB/s vs.  " << 9/time << " GB/s " << runtime << " (" << ftime/time * 100 << "% speedup overall, " << ftime/runtime * 100 << "% speedup runtime)\n";
}
