#include <openacc.h>
#include <stdlib.h>
#include <stdio.h>

#define GANGS   5
#define WORKERS 10
#define VECTOR_WIDTH    32
#define TOTAL_THREADS ((GANGS) * (WORKERS) * (VECTOR_WIDTH))

int main(int argc, char **argv) {

    int sum = 0;
#pragma acc parallel num_gangs(GANGS) num_workers(WORKERS) \
        vector_length(VECTOR_WIDTH)
    {
#pragma acc loop gang worker vector
        for (int i = 0; i < TOTAL_THREADS; i++) {
#pragma acc atomic update
            sum = sum + 1;
        }
    }

    printf("sum=%d, expected %d\n", sum, TOTAL_THREADS);

    return 0;
}
