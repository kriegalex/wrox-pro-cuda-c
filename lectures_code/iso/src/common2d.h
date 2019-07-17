#ifndef COMMON_2D_H
#define COMMON_2D_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "common.h"

typedef struct _config {
    int nx, ny;
    int nsteps;
    int save_text;
    int verbose;
    int radius;
    int ngpus;

    source *srcs;
    int nsrcs;

    int progress_width;
    int progress_disabled;
} config;

/*
 * Allow different applications to define their custom format through
 * POINT_OFFSET.
 */
#ifdef PADDING

#define POINT_OFFSET(x, y, dimx, radius) \
    (((radius) + (y)) * (dimx) + ((TRANSACTION_LEN) + (x)))

#else

#define POINT_OFFSET(x, y, dimx, radius) \
    (((radius) + (y)) * (dimx) + ((radius) + (x)))

#endif

extern void save_text(TYPE *field, const int dimx, const int dimy,
        const int ny, const int nx, const char *filename, int radius);
extern void init_data(TYPE *curr, TYPE *next, TYPE *vsq,
                TYPE *h_coeff, const int dimx, const int dimy,
                const TYPE dx, const TYPE dt);
extern void usage(char **argv);
extern void default_config(config *conf);
extern void setup_config(config *conf, int argc, char **argv);

#endif // COMMON_2D_H
