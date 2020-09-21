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

int Network::next_batch()
{
    uintmax_t lines = 0;
    while(size_t bytes_read = read(data, buf, BUFFER_SIZE))
    {
        if (!bytes_read) break;
        for(char *p = buf; lines < 10;) {
            char* bound = (char*) memchr(p, '\n', (buf + bytes_read) - p);
            if (bound - p < 0) break; // Stop.
            for (int i=0; i<layers[0].contents->cols(); ++i) (*layers[0].contents)(lines,i) = scan(&p);
            (*labels)(lines,0) = scan(&p);
            p = bound + 1;
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
