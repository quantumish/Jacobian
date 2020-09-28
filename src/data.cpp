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

void prep(char* rname, char* wname)
{
    FILE* wptr = fopen(wname, "wb");
    int fd = open(rname, O_RDONLY | O_NONBLOCK);
    if(fd == -1) throw std::runtime_error{"prep() could not open file for binary translation."};
    float tmp;
    char buf[BUFFER_SIZE+1];
    while(size_t bytes_read = read(fd, buf, BUFFER_SIZE)) {
        if(bytes_read == (size_t)-1) {
            printf("bytes_read == (size_t)-1\n");
        } 
        if (!bytes_read) break;
        for(char* p = buf;;) {
            char* bound = static_cast<char*>(memchr(p, '\n', (buf + bytes_read) - p));
            if (bound - p < 0) break; // Stop.
            for (int i=0; i<5; ++i) {
                tmp = scan(&p);
                fwrite(static_cast<void*>(&tmp), sizeof(float), 1, wptr);
            }
            p = bound + 1;
        }
    }
}

int Network::next_batch(int fd)
{
    Expects(fd > 0); // File descriptor must be valid.
    uintmax_t lines = 0;
    while(size_t bytes_read = read(fd, buf, BUFFER_SIZE)) {
        printf("hi\n");
        if (!bytes_read) {
            printf("hit end of file\n");
            break;
        }
        for(std::byte* p = buf; p < buf+BUFFER_SIZE;) {
            if (lines >= 10) return 0;
            for (int i=0; i<layers[0].contents->cols(); ++i) {
                printf("%llx %f\n", *((int*)p), *(reinterpret_cast<float*>(p)));
                (*layers[0].contents)(lines,i) = *(reinterpret_cast<float*>(p));
                p += sizeof(float);
            }
            printf("%f\n", *(reinterpret_cast<float*>(p)));
            (*labels)(lines,0) = *(reinterpret_cast<float*>(p));
            p += sizeof(float);
            ++lines;
        }
    }
}

int prep_file(char* path, char* out_path)
{
    FILE* rptr = fopen(path, "r");
    if (!rptr) throw std::runtime_error{"prep_file() could not open file for shuffle/read."};
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
    if (!src) throw std::runtime_error{"split_file() could not open file to split."};
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
