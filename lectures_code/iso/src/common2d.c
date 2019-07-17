#include "common2d.h"

/*
 * Save a single layer of the output matrix to an output text file. This can be
 * used in conjunction with iso.gnu to visualize the output.
 */
void save_text(TYPE *field, const int dimx, const int dimy,
        const int ny, const int nx, const char *filename, int radius) {
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open output file %s\n", filename);
        exit(1);
    }

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            fprintf(fp, "%d %d %.20f\n", y, x,
                    field[POINT_OFFSET(x, y, dimx, radius)]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

void init_data(TYPE *curr, TYPE *next, TYPE *vsq,
                TYPE *h_coeff, const int dimx, const int dimy,
                const TYPE dx, const TYPE dt) {
    // init the vsq array -------------------------
    for (size_t i = 0; i < dimx * dimy; i++) {
        vsq[i] = 2500. * 2500. * dt * dt; // velocity constant at 2.5 km/s
    }
    
    // init the pressure arrays -------------------
    for (size_t i = 0; i < dimx * dimy; i++) {
        curr[i] = next[i] = 0;
    }

    memset(h_coeff, 0, NUM_COEFF * sizeof(TYPE));
    TYPE scale = 1. / (dx * dx);
    h_coeff[0] = -8.541666 * scale;
    h_coeff[1] =  1.600000 * scale;
    h_coeff[2] = -0.200000 * scale;
    h_coeff[3] =  0.025397 * scale;
    h_coeff[4] = -0.001785 * scale;
}

void usage(char **argv) {
    fprintf(stderr, "usage: %s [-v] [-x nx] [-y ny] [-i iters] "
            "[-t text] [-p x,y,f] [-r radius] [-g ngpus] [-w progress_width]\n",
            argv[0]);
    exit(1);
}

void default_config(config *conf) {
    conf->nx = 2 * 256;
    conf->ny = 2 * 256;
    conf->nsteps = 100;
    conf->save_text = 0;
    conf->verbose = 0;
    conf->radius = 4;
    conf->ngpus = NO_GPU_SET;

    conf->srcs = NULL;
    conf->nsrcs = 0;

    conf->progress_width = 80;
    conf->progress_disabled = 0;
}

void setup_config(config *conf, int argc, char **argv) {
    int c;
    opterr = 0;

    default_config(conf);

    while ((c = getopt(argc, argv, "x:y:z:i:svr:tp:g:w:d")) != -1) {
        switch (c) {
            case 'x':
                conf->nx = atoi(optarg);
                break;
            case 'y':
                conf->ny = atoi(optarg);
                break;
            case 'i':
                conf->nsteps = atoi(optarg);
                break;
            case 'v':
                conf->verbose = 1;
                break;
            case 't':
                conf->save_text = 1;
                break;
            case 'p':
                conf->srcs = (source *)realloc(conf->srcs, sizeof(source) *
                        (conf->nsrcs + 1));
                parse_source(optarg, conf->srcs + conf->nsrcs);
                conf->nsrcs++;
                break;
            case 'r':
                conf->radius = atoi(optarg);
                break;
            case 'g':
                conf->ngpus = atoi(optarg);
                break;
            case 'w':
                conf->progress_width = atoi(optarg);
                break;
            case 'd':
                conf->progress_disabled = 1;
                break;
            case '?':
                fprintf(stderr, "Missing argument to option %c\n", optopt);
            default:
                usage(argv);
        }
    }

    if (conf->ngpus == NO_GPU_SET) {
        conf->ngpus = getNumCUDADevices();
    }
}
