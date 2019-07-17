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
#include "common2d.h"

#define BDIMX   32
#define BDIMY   16
#define SHAREDX(radius) (BDIMX + 2 * (radius))
#define SHAREDY(radius) (BDIMY + 2 * (radius))
#define CACHE_INDEX(y, x, radius)   ((y) * SHAREDX(radius) + (x))

#ifndef PERC_CPU
#define PERC_CPU    0.5
#endif

__constant__ TYPE const_c_coeff[NUM_COEFF];

__global__ void fwd_kernel(TYPE *next, TYPE *curr, TYPE *vsq,
        int nx, int ny, int dimx, int radius) {
    extern __shared__ TYPE cache[];

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int this_offset = POINT_OFFSET(x, y, dimx, radius);
    const int this_y = radius + threadIdx.y;
    const int this_x = radius + threadIdx.x;

    cache[CACHE_INDEX(this_y, this_x, radius)] =
        curr[POINT_OFFSET(x, y, dimx, radius)];
    if (threadIdx.y < radius) {
        cache[CACHE_INDEX(threadIdx.y, this_x, radius)] =
            curr[POINT_OFFSET(x, y - radius, dimx, radius)];
    }
    if (threadIdx.y >= radius && threadIdx.y < 2 * radius) {
        cache[CACHE_INDEX(threadIdx.y + blockDim.y, this_x, radius)] =
            curr[POINT_OFFSET(x, y - radius + blockDim.y, dimx, radius)];
    }
    if (threadIdx.x < radius) {
        cache[CACHE_INDEX(this_y, threadIdx.x, radius)] =
            curr[POINT_OFFSET(x - radius, y, dimx, radius)];
    }
    if (threadIdx.x >= radius && threadIdx.x < 2 * radius) {
        cache[CACHE_INDEX(this_y, threadIdx.x + blockDim.x, radius)] =
            curr[POINT_OFFSET(x - radius + blockDim.x, y, dimx, radius)];
    }

    __syncthreads();

    TYPE temp = 2.0f * cache[CACHE_INDEX(this_y, this_x, radius)] - next[this_offset];
    TYPE div = const_c_coeff[0] * cache[CACHE_INDEX(this_y, this_x, radius)];

    for (int d = radius; d >= 1; d--) {
        div += const_c_coeff[d] * (cache[CACHE_INDEX(this_y + d, this_x, radius)] +
                cache[CACHE_INDEX(this_y - d, this_x, radius)] + cache[CACHE_INDEX(this_y, this_x + d, radius)] +
                cache[CACHE_INDEX(this_y, this_x - d, radius)]);
    }
    next[this_offset] = temp + div * vsq[this_offset];
}

static void fwd(TYPE *next, TYPE *curr, TYPE *vsq,
        TYPE *c_coeff, int nx, int ny, int dimx, int radius) {

#pragma omp parallel for collapse(2)
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            int this_offset = POINT_OFFSET(x, y, dimx, radius);
            TYPE temp = 2.0f * curr[this_offset] - next[this_offset];
            TYPE div = c_coeff[0] * curr[this_offset];
            for (int d = 1; d <= radius; d++) {
                int y_pos_offset = POINT_OFFSET(x, y + d, dimx, radius);
                int y_neg_offset = POINT_OFFSET(x, y - d, dimx, radius);
                int x_pos_offset = POINT_OFFSET(x + d, y, dimx, radius);
                int x_neg_offset = POINT_OFFSET(x - d, y, dimx, radius);
                div += c_coeff[d] * (curr[y_pos_offset] +
                        curr[y_neg_offset] + curr[x_pos_offset] +
                        curr[x_neg_offset]);
            }
            next[this_offset] = temp + div * vsq[this_offset];
        }
    }
}

int main(int argc, char *argv[]) {
    config conf;
    setup_config(&conf, argc, argv);
    init_progress(conf.progress_width, conf.nsteps, conf.progress_disabled);

#ifndef PADDING
    fprintf(stderr, "Must be compiled with -DPADDING\n");
    return 1;
#endif

    if (conf.nx % BDIMX != 0) {
        fprintf(stderr, "Invalid nx configuration, must be an even multiple of "
                "%d\n", BDIMX);
        return 1;
    }
    if (conf.ny % BDIMY != 0) {
        fprintf(stderr, "Invalid ny configuration, must be an even multiple of "
                "%d\n", BDIMY);
        return 1;
    }
    if (conf.radius > TRANSACTION_LEN) {
        fprintf(stderr, "Radius must be less than TRANSACTION_LEN to include "
                "it in dimx padding\n");
        return 1;
    }

    TYPE dx = 20.f;
    TYPE dt = 0.002f;

    // compute the pitch for perfect coalescing
    size_t dimx = TRANSACTION_LEN + conf.nx + conf.radius;
    dimx += (TRANSACTION_LEN - (dimx % TRANSACTION_LEN));
    size_t dimy = conf.ny + 2*conf.radius;
    size_t nbytes = dimx * dimy * sizeof(TYPE);

    if (conf.verbose) {
        printf("x = %zu, y = %zu\n", dimx, dimy);
        printf("nsteps = %d\n", conf.nsteps);
        printf("radius = %d\n", conf.radius);
    }

    TYPE c_coeff[NUM_COEFF];
    TYPE *curr, *next, *vsq;
    CHECK(cudaMallocHost((void **)&curr, nbytes));
    CHECK(cudaMallocHost((void **)&next, nbytes));
    CHECK(cudaMallocHost((void **)&vsq, nbytes));

    config_sources(&conf.srcs, &conf.nsrcs, conf.nx, conf.ny, conf.nsteps);
    TYPE **srcs = sample_sources(conf.srcs, conf.nsrcs, conf.nsteps, dt);

    init_data(curr, next, vsq, c_coeff, dimx, dimy, dx, dt);

    int ny_on_cpu = PERC_CPU * conf.ny;
    int ny_on_gpu = conf.ny - ny_on_cpu;
    printf("Processing %d layers on the GPU, %d on the CPU\n", ny_on_gpu,
            ny_on_cpu);

    TYPE *d_curr, *d_next, *d_vsq;
    CHECK(cudaMalloc((void **)&d_curr, nbytes));
    CHECK(cudaMalloc((void **)&d_next, nbytes));
    CHECK(cudaMalloc((void **)&d_vsq, nbytes));

    dim3 block(BDIMX, BDIMY);
    dim3 grid(conf.nx / block.x, conf.ny / block.y);

    double mem_start = seconds();

    CHECK(cudaMemcpy(d_curr, curr, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_next, next, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_vsq, vsq, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(const_c_coeff, c_coeff, NUM_COEFF * sizeof(TYPE)));
    double start = seconds();
    for (int step = 0; step < conf.nsteps; step++) {
        for (int src = 0; src < conf.nsrcs; src++) {
            if (conf.srcs[src].t > step) continue;
            int src_offset = POINT_OFFSET(conf.srcs[src].x, conf.srcs[src].y,
                    dimx, conf.radius);
            CHECK(cudaMemcpy(d_curr + src_offset, srcs[src] + step,
                        sizeof(TYPE), cudaMemcpyHostToDevice));
            curr[src_offset] = srcs[src][step];
        }

        fwd_kernel<<<grid, block, SHAREDY(conf.radius) * SHAREDX(conf.radius) *
            sizeof(TYPE)>>>(d_next, d_curr, d_vsq, conf.nx, conf.ny, dimx, conf.radius);
        fwd(next + (ny_on_gpu * dimx), curr + (ny_on_gpu * dimx),
                vsq + (ny_on_gpu * dimx), c_coeff, conf.nx, ny_on_cpu, dimx, conf.radius);

        CHECK(cudaMemcpy(next + (ny_on_gpu * dimx), d_next + (ny_on_gpu * dimx),
                    conf.radius * dimx * sizeof(TYPE), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(d_next + ((ny_on_gpu + conf.radius) * dimx),
                    next + ((ny_on_gpu + conf.radius) * dimx),
                    conf.radius * dimx * sizeof(TYPE), cudaMemcpyHostToDevice));

        TYPE *tmp = d_next;
        d_next = d_curr;
        d_curr = tmp;

        tmp = next;
        next = curr;
        curr = tmp;

        update_progress(step + 1);
    }
    CHECK(cudaDeviceSynchronize());
    double compute_s = seconds() - start;

    CHECK(cudaMemcpy(curr, d_curr, (conf.radius + ny_on_gpu) * dimx * sizeof(TYPE),
                cudaMemcpyDeviceToHost));
    double total_s = seconds() - mem_start;

    finish_progress();

    float point_rate = (float)conf.nx * conf.ny / (compute_s / conf.nsteps);
    fprintf(stderr, "iso_r4_2x:   %8.10f s total, %8.10f s/step, %8.2f Mcells/s/step\n",
            total_s, compute_s / conf.nsteps, point_rate / 1000000.f);

    if (conf.save_text) {
        save_text(curr, dimx, dimy, conf.ny, conf.nx,
                 "snap.text", conf.radius);
    }

    CHECK(cudaFreeHost(curr));
    CHECK(cudaFreeHost(next));
    CHECK(cudaFreeHost(vsq));
    for (int i = 0; i < conf.nsrcs; i++) {
        free(srcs[i]);
    }
    free(srcs);

    CHECK(cudaFree(d_curr));
    CHECK(cudaFree(d_next));
    CHECK(cudaFree(d_vsq));

    return 0;
}
