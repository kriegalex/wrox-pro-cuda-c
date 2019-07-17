#include <openacc.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

#define LOOP    30

static void usage(char **argv) {
    fprintf(stderr, "usage: %s [-n N] [-m M] [-c] [-g gangs] [-v vector_length] [-h]\n", argv[0]);
    exit(1);
}

static void verify(int *A, int *B, int *C, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int index = i * N + j;
            if (C[index] != A[index] + B[index]) {
                fprintf(stderr, "Validation error on element (%d, %d). "
                    "Expected %d, got %d\n", i, j, A[index] + B[index],
                    C[index]);
                exit(1);
            }
        }
    }
}

int main(int argc, char **argv) {
    int check = 0;
    int gangs = 15;
    int vector_length = 16;
    int M = 1024;
    int N = 1024;
    int c;
    opterr = 0;

    while ((c = getopt(argc, argv, "n:m:cg:v:h")) != -1) {
        switch (c) {
            case 'n':
                N = atoi(optarg);
                break;
            case 'm':
                M = atoi(optarg);
                break;
            case 'c':
                check = 1;
                break;
            case 'g':
                gangs = atoi(optarg);
                break;
            case 'v':
                vector_length = atoi(optarg);
                break;
            case '?':
                fprintf(stderr, "Missing argument to option %c\n", optopt);
            case 'h':
            default:
                usage(argv);
        }
    }

    int *restrict A = (int *)malloc(M * N * sizeof(int));
    int *restrict B = (int *)malloc(M * N * sizeof(int));
    int *restrict C = (int *)malloc(M * N * sizeof(int));

    for (int i = 0; i < M * N; i++) {
        A[i] = i / N;
        B[i] = i % N;
    }

    // warmup
#pragma acc kernels loop copyin(A[0:M * N], B[0:M * N]) copyout(C[0:M * N])
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int index = i * N + j;
            C[index] = A[index] + B[index];
        }
    }

    printf("Using %d gangs, vector length of %d\n", gangs, vector_length);

    for (int l = 0; l < LOOP; l++) {
#pragma acc parallel copyin(A[0:M * N], B[0:M * N]) copyout(C[0:M * N])
#pragma acc loop gang(gangs)
        for (int i = 0; i < M; i++) {
#pragma acc loop vector(vector_length)
            for (int j = 0; j < N; j++) {
                int index = i * N + j;
                C[index] = A[index] + B[index];
            }
        }

        if (check) {
            verify(A, B, C, M, N);
        }
    }

    return 0;
}
