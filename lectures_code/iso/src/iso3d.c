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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include "common.h"
#include "common3d.h"

/*
 * This function advances the state of the system by nsteps timesteps. The 
 * curr is the current state of the system.
 * next is the output matrix to store the next time step into.
 */
static void fwd(TYPE *next, TYPE *curr, TYPE *vsq,
        TYPE *c_coeff, int nx, int ny, int nz, int dimx, int dimy, int dimz,
        int radius) {

#pragma omp parallel for collapse(3)
    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                int this_offset = POINT_OFFSET(x, y, z, dimy, dimx, radius);
                TYPE temp = 2.0f * curr[this_offset] - next[this_offset];
                TYPE div = c_coeff[0] * curr[this_offset];
                for (int d = 1; d <= radius; d++) {
                    int z_pos_offset = POINT_OFFSET(x, y, z + d, dimy, dimx,
                            radius);
                    int z_neg_offset = POINT_OFFSET(x, y, z - d, dimy, dimx,
                            radius);
                    int y_pos_offset = POINT_OFFSET(x, y + d, z, dimy, dimx,
                            radius);
                    int y_neg_offset = POINT_OFFSET(x, y - d, z, dimy, dimx,
                            radius);
                    int x_pos_offset = POINT_OFFSET(x + d, y, z, dimy, dimx,
                            radius);
                    int x_neg_offset = POINT_OFFSET(x - d, y, z, dimy, dimx,
                            radius);
                    div += c_coeff[d] * (curr[z_pos_offset] +
                            curr[z_neg_offset] + curr[y_pos_offset] +
                            curr[y_neg_offset] + curr[x_pos_offset] +
                            curr[x_neg_offset]);
                }
                next[this_offset] = temp + div * vsq[this_offset];
            }
        }
    }
}

int main( int argc, char *argv[] ) {
    config conf;
    setup_config(&conf, argc, argv);
    init_progress(conf.progress_width, conf.nsteps, conf.progress_disabled);

    TYPE dx = 20.f;
    TYPE dt = 0.002f;

    // compute the pitch for perfect coalescing
    size_t dimx = conf.nx + 2*conf.radius;
    size_t dimy = conf.ny + 2*conf.radius;
    size_t dimz = conf.nz + 2*conf.radius;
    size_t nbytes = dimx * dimy * dimz * sizeof(TYPE);

    if (conf.verbose) {
        printf("x = %zu, y = %zu, z = %zu\n", dimx, dimy, dimz);
        printf("nsteps = %d\n", conf.nsteps);
        printf("radius = %d\n", conf.radius);
    }

    TYPE c_coeff[NUM_COEFF];
    TYPE *curr = (TYPE *)malloc(nbytes);
    TYPE *next = (TYPE *)malloc(nbytes);
    TYPE *vsq  = (TYPE *)malloc(nbytes);
    if (curr == NULL || next == NULL || vsq == NULL) {
        fprintf(stderr, "Allocations failed\n");
        return 1;
    }

    config_sources(&conf.srcs, &conf.nsrcs, conf.nx, conf.ny, conf.nsteps);
    TYPE **srcs = sample_sources(conf.srcs, conf.nsrcs, conf.nsteps, dt);

    init_data(curr, next, vsq, c_coeff, dimx, dimy, dimz, dx, dt);

    double start = seconds();
    for (int step = 0; step < conf.nsteps; step++) {
        for (int src = 0; src < conf.nsrcs; src++) {
            if (conf.srcs[src].t > step) continue;
            int src_offset = POINT_OFFSET(conf.srcs[src].x, conf.srcs[src].y, 0,
                    dimy, dimx, conf.radius);
            curr[src_offset] = srcs[src][step];
        }

        fwd(next, curr, vsq, c_coeff, conf.nx, conf.ny, conf.nz,
                dimx, dimy, dimz, conf.radius);

        TYPE *tmp = next;
        next = curr;
        curr = tmp;

        update_progress(step + 1);
    }
    double elapsed_s = seconds() - start;

    finish_progress();

    float point_rate = (float)conf.nx * conf.ny / (elapsed_s / conf.nsteps);
    fprintf(stderr, "iso_r4_2x:   %8.10f s total, %8.10f s/step, %8.2f Mcells/s/step\n",
            elapsed_s, elapsed_s / conf.nsteps, point_rate / 1000000.f);

    if (conf.save_text != -1) {
        save_layer_text(curr, conf.save_text, dimx, dimy, conf.ny, conf.nx,
                "snap.text", conf.radius);
    }

    free(curr);
    free(next);
    free(vsq);
    for (int i = 0; i < conf.nsrcs; i++) {
        free(srcs[i]);
    }
    free(srcs);

    return 0;
}
