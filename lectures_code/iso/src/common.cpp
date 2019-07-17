/*
 * Copyright (c) 2012, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived 
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Paulius Micikevicius (pauliusm@nvidia.com)
 * Max Grossman (jmaxg3@gmail.com)
 */
#include <sys/time.h>
#include "common.h"
#include <math.h>

static char *progress_buffer = NULL;
static int progress_length = -1;
static int progress_goal = -1;
static int progress_num_ticks = -1;
static int progress_disabled = 0;

double seconds() {
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void ricker_wavelet(TYPE *source, int nsteps, TYPE dt, TYPE freq) {
    TYPE shift = -1.5594f / freq;

    for (int i = 0; i < nsteps; i++) {
        TYPE time = i*dt + shift;
        TYPE pi_freq_t = 3.141517f * freq * time;
        TYPE sqr_pi_freq_t = pi_freq_t * pi_freq_t;
        source[i] = 1e5f * (1.f - 2 * sqr_pi_freq_t) * exp(-sqr_pi_freq_t);
    }
}

void parse_source(char *optarg, source *out) {
    char *x_str = optarg;
    char *first_comma = strchr(x_str, ',');
    if (first_comma == NULL) {
        fprintf(stderr, "Improperly formatted argument to -p, must "
                "be x,y,f,t\n");
        exit(1);
    }
    char *y_str = first_comma + 1;
    char *second_comma = strchr(y_str, ',');
    if (second_comma == NULL) {
        fprintf(stderr, "Improperly formatted argument to -p, must "
                "be x,y,f,t\n");
        exit(1);
    }
    char *freq_str = second_comma + 1;
    char *third_comma = strchr(freq_str, ',');
    if (third_comma == NULL) {
        fprintf(stderr, "Improperly formatted argument to -p, must "
                "be x,y,f,t\n");
        exit(1);
    }
    char *time_str = third_comma + 1;
    *first_comma = '\0';
    *second_comma = '\0';
    *third_comma = '\0';

    out->x = atoi(x_str);
    out->y = atoi(y_str);
    out->freq = atof(freq_str);
    out->t = atoi(time_str);
}

void config_sources(source **srcs, int *nsrcs, int nx, int ny, int nsteps) {
    if (*nsrcs == 0) {
        *srcs = (source *)malloc(sizeof(source));
        if (*srcs == NULL) {
            fprintf(stderr, "Allocation failed\n");
            exit(1);
        }
        (*srcs)->x = nx / 2;
        (*srcs)->y = ny / 2;
        (*srcs)->freq = 15.0f;
        (*srcs)->t = 0;
        *nsrcs = 1;
    }

    // Validate sources
    for (int i = 0; i < *nsrcs; i++) {
        source *curr = (*srcs) + i;
        if (curr->x < 0 || curr->x >= nx) {
            fprintf(stderr, "Invalid x value for source\n");
            exit(1);
        }
        if (curr->y < 0 || curr->y >= ny) {
            fprintf(stderr, "Invalid y value for source\n");
            exit(1);
        }
        if (curr->t < 0 || curr->t >= nsteps) {
            fprintf(stderr, "Invalid t value for source\n");
            exit(1);
        }
    }
}

TYPE **sample_sources(source *srcs, int nsrcs, int nsteps, TYPE dt) {
    TYPE **src_samples = (TYPE **)malloc(nsrcs * sizeof(TYPE *));
    if (src_samples == NULL) {
        fprintf(stderr, "Allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < nsrcs; i++) {
        src_samples[i] = (TYPE *)malloc(nsteps * sizeof(TYPE));
        if (src_samples[i] == NULL) {
            fprintf(stderr, "Allocation failed\n");
            exit(1);
        }
        ricker_wavelet(src_samples[i], nsteps, dt, srcs[i].freq);
    }
    return src_samples;
}

void init_progress(int length, int goal, int disabled) {
    int i;
    if (progress_buffer != NULL) {
        fprintf(stderr, "Progress initialized multiple times\n");
        exit(1);
    }

    if (length > 100) {
        fprintf(stderr, "Invalid progress length, must be <= 100\n");
        exit(1);
    }

    progress_disabled = disabled;

    if (disabled) return;

    progress_buffer = (char *)malloc(sizeof(char) * length);
    progress_length = length;
    progress_goal = goal;
    progress_num_ticks = 0;

    progress_buffer[0] = '|';
    progress_buffer[length + 1] = '|';
    progress_buffer[length + 2] = '\0';

    for (i = 1; i <= length; i++) {
        progress_buffer[i] = '-';
    }

    fprintf(stderr, "%s", progress_buffer);
}

void update_progress(int progress) {
    int i;

    if (progress_disabled) {
        return;
    }

    if (progress_buffer == NULL) {
        fprintf(stderr, "Calling update_progress without having called "
                "init_progress\n");
        exit(1);
    }

    double perc_progress = (double)progress / (double)progress_goal;
    int ticks = (int)(perc_progress * progress_length);
    if (ticks > progress_length) {
        ticks = progress_length;
    }

    if (ticks < progress_num_ticks) {
        fprintf(stderr, "Ticks went backwards?\n");
        exit(1);
    }

    if (ticks > progress_num_ticks) {
        for (i = 1; i < 1 + ticks; i++) {
            progress_buffer[i] = '=';
        }

        for (i = 0; i < progress_length + 2; i++) {
            fprintf(stderr, "\b");
        }

        fprintf(stderr, "%s", progress_buffer);
    }

    progress_num_ticks = ticks;
}

void finish_progress() {
    if (progress_disabled) {
        return;
    }

    update_progress(progress_goal);
    fprintf(stderr, "\n");

    free(progress_buffer);
}


