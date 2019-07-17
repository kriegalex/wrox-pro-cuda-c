#ifndef COMMON_H
#define COMMON_H

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

#define NUM_COEFF   20
#define NO_GPU_SET  -1

#ifndef TYPE
#define TYPE    float
#endif

#define TAPER_WIDTH 20
#define TRANSACTION_LEN (128 / sizeof(TYPE))

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "common_cuda.h"

typedef struct _source {
    int x, y;
    float freq;
    int t;
} source;

extern double seconds();
extern void ricker_wavelet(TYPE *source, int nsteps, TYPE dt, TYPE freq);
extern void parse_source(char *optarg, source *out);
extern void config_sources(source **srcs, int *nsrcs, int dimx, int dimy,
        int nsteps);
extern TYPE **sample_sources(source *srcs, int nsrcs, int nsteps, TYPE dt);

extern void init_progress(int length, int goal, int disabled);
extern void update_progress(int progress);
extern void finish_progress();

#endif // COMMON_H
