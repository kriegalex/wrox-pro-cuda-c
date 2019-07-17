#include <openacc.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#define LOOP    30
#define BAD_STRIDE  4

static void usage(char **argv) {
    fprintf(stderr, "usage: %s [-n size] [-h] [-c] [-b]\n", argv[0]);
    exit(1);
}

static void verify(int *A, int *B, int *C, int N) {
    for (int i = 0; i < N; i++) {
        if (C[i] != A[i] + B[i]) {
            fprintf(stderr, "Validation error on element %d. Expected %d, got "
                "%d\n", i, A[i] + B[i], C[i]);
            exit(1);
        }
    }
}

int main(int argc, char **argv) {
    int check = 0;
    int bad = 0;
    int N = 1024 * 1024;
    int c;
    opterr = 0;

    while ((c = getopt(argc, argv, "bcn:h")) != -1) {
        switch (c) {
            case 'n':
                N = atoi(optarg);
                break;
            case 'c':
                check = 1;
                break;
            case 'b':
                bad = 1;
                break;
            case '?':
                fprintf(stderr, "Missing argument to option %c\n", optopt);
            case 'h':
            default:
                usage(argv);
        }
    }

    int *restrict A = (int *)malloc(N * sizeof(int));
    int *restrict B = (int *)malloc(N * sizeof(int));
    int *restrict C = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    // warmup
#pragma acc kernels loop copyin(A[0:N], B[0:N]) copyout(C[0:N])
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }

    if (!bad) {
        for (int l = 0; l < LOOP; l++) {
#pragma acc kernels loop copyin(A[0:N], B[0:N]) copyout(C[0:N])
            for (int i = 0; i < N; i++) {
                C[i] = A[i] + B[i];
            }
            if (check) verify(A, B, C, N);
        }
    } else {
        for (int l = 0; l < LOOP; l++) {
#pragma acc kernels loop copyin(A[0:N], B[0:N]) copyout(C[0:N])
            for (int i = 0; i < N; i++) {
                int round = i / (N / BAD_STRIDE);
                int offset = i % (N / BAD_STRIDE);
                C[offset * BAD_STRIDE + round] = A[offset * BAD_STRIDE + round] +
                    B[offset * BAD_STRIDE + round];
            }
            if (check) verify(A, B, C, N);
        }
    }

    return 0;
}
