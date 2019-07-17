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

#define LOW_HALO_START_Y(gpu, ny_per_gpu) \
    ((gpu) * (ny_per_gpu))

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

    if (conf.ny % conf.ngpus != 0) {
        fprintf(stderr, "ny must be evenly divisible by ngpus (%d)\n",
                conf.ngpus);
        return 1;
    }
    int ny_per_gpu = conf.ny / conf.ngpus;

    TYPE dx = 20.f;
    TYPE dt = 0.002f;

    // compute the pitch for perfect coalescing
    size_t dimx = TRANSACTION_LEN + conf.nx + conf.radius;
    dimx += (TRANSACTION_LEN - (dimx % TRANSACTION_LEN));
    size_t dimy_total = conf.ny + 2 * conf.radius;
    size_t dimy_per_gpu = ny_per_gpu + 2 * conf.radius;
    size_t nbytes_total = dimx * dimy_total * sizeof(TYPE);
    size_t nbytes_per_gpu = dimx * dimy_per_gpu * sizeof(TYPE);

    if (conf.verbose) {
        printf("x = %zu, y = %zu\n", dimx, dimy_total);
        printf("nsteps = %d\n", conf.nsteps);
        printf("radius = %d\n", conf.radius);
    }

    TYPE c_coeff[NUM_COEFF];
    TYPE *curr, *next, *vsq;
    CHECK(cudaMallocHost((void **)&curr, nbytes_total));
    CHECK(cudaMallocHost((void **)&next, nbytes_total));
    CHECK(cudaMallocHost((void **)&vsq, nbytes_total));

    config_sources(&conf.srcs, &conf.nsrcs, conf.nx, conf.ny, conf.nsteps);
    TYPE **srcs = sample_sources(conf.srcs, conf.nsrcs, conf.nsteps, dt);

    init_data(curr, next, vsq, c_coeff, dimx, dimy_total, dx, dt);

    TYPE **d_curr, **d_next, **d_vsq;
    d_curr = (TYPE **)malloc(sizeof(TYPE *) * conf.ngpus);
    d_next = (TYPE **)malloc(sizeof(TYPE *) * conf.ngpus);
    d_vsq = (TYPE **)malloc(sizeof(TYPE *) * conf.ngpus);
    cudaStream_t *streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) *
            conf.ngpus);
    for (int g = 0; g < conf.ngpus; g++) {
        CHECK(cudaSetDevice(g));
        CHECK(cudaMalloc((void **)(d_curr + g), nbytes_per_gpu));
        CHECK(cudaMalloc((void **)(d_next + g), nbytes_per_gpu));
        CHECK(cudaMalloc((void **)(d_vsq + g), nbytes_per_gpu));
        CHECK(cudaStreamCreate(streams + g));
    }

    dim3 block(BDIMX, BDIMY);
    dim3 grid(conf.nx / block.x, ny_per_gpu / block.y);

    double mem_start = seconds();

    for (int g = 0; g < conf.ngpus; g++) {
        int low_halo_start = LOW_HALO_START_Y(g, ny_per_gpu);
        CHECK(cudaSetDevice(g));

        CHECK(cudaMemcpyToSymbol(const_c_coeff, c_coeff,
                    NUM_COEFF * sizeof(TYPE)));

        CHECK(cudaMemcpyAsync(d_curr[g],
                    curr + (low_halo_start * dimx), nbytes_per_gpu,
                    cudaMemcpyHostToDevice, streams[g]));
        CHECK(cudaMemcpyAsync(d_next[g],
                    next + (low_halo_start * dimx), nbytes_per_gpu,
                    cudaMemcpyHostToDevice, streams[g]));
        CHECK(cudaMemcpyAsync(d_vsq[g],
                    vsq + (low_halo_start * dimx), nbytes_per_gpu,
                    cudaMemcpyHostToDevice, streams[g]));
    }
    double start = seconds();
    for (int step = 0; step < conf.nsteps; step++) {
        for (int src = 0; src < conf.nsrcs; src++) {
            if (conf.srcs[src].t > step) continue;

            int src_y = conf.srcs[src].y;
            for (int g = 0; g < conf.ngpus; g++) {
                // will be negative for first GPU
                int base_y = (g * ny_per_gpu) - conf.radius;
                int limit_y = ((g + 1) * ny_per_gpu) + conf.radius;
                if (src_y >= base_y && src_y < limit_y) {
                    int offset_y = src_y - base_y;
                    int point_offset = (offset_y * dimx) +
                        (TRANSACTION_LEN + conf.srcs[src].x);
                    CHECK(cudaMemcpyAsync(d_curr[g] + point_offset,
                                srcs[src] + step, sizeof(TYPE),
                                cudaMemcpyHostToDevice, streams[g]));
                }
            }
        }

        for (int g = 0; g < conf.ngpus; g++) {
            CHECK(cudaSetDevice(g));
            fwd_kernel<<<grid, block, SHAREDY(conf.radius) * SHAREDX(conf.radius) *
                sizeof(TYPE), streams[g]>>>(d_next[g], d_curr[g],
                        d_vsq[g], conf.nx, ny_per_gpu, dimx, conf.radius);
        }

        if (step < conf.nsteps - 1) {
            for (int g = 0; g < conf.ngpus; g++) {
                if (g < conf.ngpus - 1) {
                    // copy up
                    CHECK(cudaMemcpyAsync(d_next[g + 1],
                                d_next[g] + (ny_per_gpu * dimx),
                                conf.radius * dimx * sizeof(TYPE),
                                cudaMemcpyDeviceToDevice, streams[g]));
                }
                if (g > 0) {
                    // copy down
                    CHECK(cudaMemcpyAsync(
                                d_next[g - 1] + ((conf.radius + ny_per_gpu) * dimx),
                                d_next[g] + (conf.radius * dimx),
                                conf.radius * dimx * sizeof(TYPE),
                                cudaMemcpyDeviceToDevice, streams[g]));
                }
            }

            for (int g = 0; g < conf.ngpus; g++) {
                CHECK(cudaSetDevice(g));
                CHECK(cudaStreamSynchronize(streams[g]));
            }
        }

        TYPE **tmp = d_next;
        d_next = d_curr;
        d_curr = tmp;

        update_progress(step + 1);
    }

    for (int g = 0; g < conf.ngpus; g++) {
        CHECK(cudaSetDevice(g));
        CHECK(cudaDeviceSynchronize());
    }
    double compute_s = seconds() - start;

    for (int g = 0; g < conf.ngpus; g++) {
        CHECK(cudaMemcpy(curr + ((conf.radius + g * ny_per_gpu) * dimx),
                    d_curr[g] + (conf.radius * dimx),
                    ny_per_gpu * dimx * sizeof(TYPE), cudaMemcpyDeviceToHost));
    }
    double total_s = seconds() - mem_start;

    finish_progress();

    float point_rate = (float)conf.nx * conf.ny / (compute_s / conf.nsteps);
    fprintf(stderr, "iso_r4_2x:   %8.10f s total, %8.10f s/step, %8.2f Mcells/s/step\n",
            total_s, compute_s / conf.nsteps, point_rate / 1000000.f);

    if (conf.save_text) {
        save_text(curr, dimx, dimy_total, conf.ny, conf.nx,
                 "snap.text", conf.radius);
    }

    CHECK(cudaFreeHost(curr));
    CHECK(cudaFreeHost(next));
    CHECK(cudaFreeHost(vsq));
    for (int i = 0; i < conf.nsrcs; i++) {
        free(srcs[i]);
    }
    free(srcs);

    for (int g = 0; g < conf.ngpus; g++) {
        CHECK(cudaFree(d_curr[g]));
        CHECK(cudaFree(d_next[g]));
        CHECK(cudaFree(d_vsq[g]));
        CHECK(cudaStreamDestroy(streams[g]));
    }
    free(d_curr);
    free(d_next);
    free(d_vsq);
    free(streams);

    return 0;
}
