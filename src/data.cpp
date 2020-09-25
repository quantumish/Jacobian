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
    int fd = open("../../data_banknote_authentication.bin", O_RDONLY | O_NONBLOCK);
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
    int fd = open("../../data_banknote_authentication.bin", O_RDONLY | O_NONBLOCK);
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

float std_read(int BUFFER_SIZE)
{
    int fd = open("../../data_banknote_authentication.bin", O_RDONLY | O_NONBLOCK);
    fcntl(fd, F_RDADVISE);
    if(fd == -1) {


int Network::next_batch()
{
    uintmax_t lines = 0;
    while(size_t bytes_read = read(data, buf, BUFFER_SIZE))
    {
        if (!bytes_read) break;
        for(char *p = buf; p < buf+BUFFER_SIZE && lines < 10;) {
            for (int i=0; i<<layers[0].contents->cols(); ++i) {
                (*layers[0].contents)(lines,i) = *((float*)p);
                p += sizeof(float);
            }
            (*labels)(lines,0) = *((float*)p);
            p += sizeof(float);
            ++lines;
        }
    }
    // char line[MAXLINE];
    // int inputs = layers[0].contents->cols();
    // int datalen = batch_size * inputs;
    // float batch[datalen];
    // for (int i = 0; i < batch_size; i++) {
    //     fgets(line, MAXLINE, data);
    //     char *p;
    //     p = strtok(line,",");
    //     for (int j = 0; j < inputs; j++) {
    //         batch[j + (i * inputs)] = strtod(p, NULL);
    //         p = strtok(NULL,",");
    //     }
    //     (*labels)(i, 0) = strtod(p, NULL);
    // }
    // float* batchptr = batch;
    // update_layer(batchptr, datalen, 0);
    // return 0;
}

int prep_file(char* path, char* out_path)
{
    FILE* rptr = fopen(path, "r");
    char line[MAXLINE];
    std::vector<std::string> lines;
    int count = 0;
    while (fgets(line, MAXLINE, rptr) != NULL) {
        lines.emplace_back(line);
        count++;
    }
    lines[lines.size()-1] = lines[lines.size()-1] + "\n";
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(lines.begin(), lines.end(), g);
    fclose(rptr);
    FILE* wptr = fopen(out_path, "w");
    for (std::string & i : lines) {
        const char* cstr = i.c_str();
        fprintf(wptr,"%s", cstr);
    }
    fclose(wptr);
    return count;
}

int split_file(char* path, int lines, float ratio)
{
    FILE* src = fopen(path, "r");
    FILE* test = fopen(VAL_PATH, "w");
    FILE* train = fopen(TRAIN_PATH, "w");
    int switch_line = round(ratio * lines);
    char line[MAXLINE];
    int tests = 0;
    for (int i = 0; fgets(line, MAXLINE, src) != NULL; i++) {
        if (i > switch_line) {
            fprintf(test, "%s", line);
            tests++;
        }
        else fprintf(train, "%s", line);
    }
    fclose(src);
    fclose(test);
    fclose(train);
    return tests;
}
