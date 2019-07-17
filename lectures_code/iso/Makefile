BINDIR = bin
SRCDIR = src
SRC = ${SRCDIR}/iso2d.c
CFLAGS=-L${BINDIR} -I${MPI_HOME}/include -L${MPI_HOME}/lib -lmpi -O2 -DTYPE=float

all: ${BINDIR}/iso2d ${BINDIR}/iso2d_omp ${BINDIR}/iso3d ${BINDIR}/iso3d_omp \
		${BINDIR}/iso2d_cuda ${BINDIR}/iso3d_cuda ${BINDIR}/iso2d_cuda_pinned \
		${BINDIR}/iso2d_cuda_global ${BINDIR}/iso2d_cuda_shared \
		${BINDIR}/iso2d_cuda_constant ${BINDIR}/iso2d_cuda_coop \
		${BINDIR}/iso2d_cuda_atomic ${BINDIR}/iso2d_cuda_unsafe \
		${BINDIR}/iso2d_cuda_multigpu ${BINDIR}/iso2d_cuda_peer \
		${BINDIR}/iso2d_cuda_zerocopy ${BINDIR}/iso2d_cuda_uva \
		${BINDIR}/iso2d_cuda_unified ${BINDIR}/iso3d_cuda_peer \
		${BINDIR}/iso3d_cuda_mpi ${BINDIR}/iso3d_cuda_mpi2 \
		${BINDIR}/iso3d_cuda_cuda-aware-mpi ${BINDIR}/iso3d_cuda_cuda-aware-mpi2 \
		${BINDIR}/iso2d_openacc_kernels ${BINDIR}/iso2d_openacc_parallel \
		${BINDIR}/iso2d_openacc_intro ${BINDIR}/iso2d_openacc_data \
		${BINDIR}/iso2d_openacc_enter_data ${BINDIR}/iso2d_openacc_runtime \
        ${BINDIR}/iso2d_openacc_multi_accel ${BINDIR}/iso2d_openacc_tile \
		${BINDIR}/iso2d_openacc_cache

${BINDIR}/iso2d_openacc_cache: ${SRCDIR}/iso2d_cache.c ${BINDIR}/libcommon_cuda.so
	pgcc ${CFLAGS} -acc -Minfo=accel ${SRCDIR}/common.c ${SRCDIR}/common2d.c -lcommon_cuda $< -o $@
${BINDIR}/iso2d_openacc_tile: ${SRCDIR}/iso2d_tile.c ${BINDIR}/libcommon_cuda.so
	pgcc ${CFLAGS} -acc -Minfo=accel ${SRCDIR}/common.c ${SRCDIR}/common2d.c -lcommon_cuda $< -o $@
${BINDIR}/iso2d_openacc_multi_accel: ${SRCDIR}/iso2d_multi_accel.c ${BINDIR}/libcommon_cuda.so
	pgc++ ${CFLAGS} -acc -Minfo=accel ${SRCDIR}/common.cpp ${SRCDIR}/common2d.cpp -lcommon_cuda $< -o $@
${BINDIR}/iso2d_openacc_runtime: ${SRCDIR}/iso2d_runtime.c ${BINDIR}/libcommon_cuda.so
	pgc++ ${CFLAGS} -acc -Minfo=accel ${SRCDIR}/common.cpp ${SRCDIR}/common2d.cpp -lcommon_cuda $< -o $@
${BINDIR}/iso2d_openacc_enter_data: ${SRCDIR}/iso2d_enter_data.c ${BINDIR}/libcommon_cuda.so
	pgc++ ${CFLAGS} -acc -Minfo=accel ${SRCDIR}/common.cpp ${SRCDIR}/common2d.cpp -lcommon_cuda $< -o $@
${BINDIR}/iso2d_openacc_data: ${SRCDIR}/iso2d_data.c ${BINDIR}/libcommon_cuda.so
	pgc++ ${CFLAGS} -acc -Minfo=accel ${SRCDIR}/common.cpp ${SRCDIR}/common2d.cpp -lcommon_cuda $< -o $@
${BINDIR}/iso2d_openacc_intro: ${SRCDIR}/iso2d_openacc-intro.c ${BINDIR}/libcommon_cuda.so
	pgc++ ${CFLAGS} -acc -Minfo=accel ${SRCDIR}/common.cpp ${SRCDIR}/common2d.cpp -lcommon_cuda $< -o $@
${BINDIR}/iso2d_openacc_parallel: ${SRCDIR}/iso2d_parallel.c ${BINDIR}/libcommon_cuda.so
	pgc++ ${CFLAGS} -acc -Minfo=accel ${SRCDIR}/common.cpp ${SRCDIR}/common2d.cpp -lcommon_cuda $< -o $@
${BINDIR}/iso2d_openacc_kernels: ${SRCDIR}/iso2d_kernels.c ${BINDIR}/libcommon_cuda.so
	pgc++ ${CFLAGS} -acc -Minfo=accel ${SRCDIR}/common.cpp ${SRCDIR}/common2d.cpp -lcommon_cuda $< -o $@

${BINDIR}/iso3d_cuda_cuda-aware-mpi2: ${SRCDIR}/iso3d_cuda-aware-mpi2.cu ${BINDIR}/libcommon_cuda.so
	nvcc ${CFLAGS} -DPADDING -arch=sm_20 ${SRCDIR}/common.cpp ${SRCDIR}/common3d.cpp -lcommon_cuda $< -o $@
${BINDIR}/iso3d_cuda_cuda-aware-mpi: ${SRCDIR}/iso3d_cuda-aware-mpi.cu ${BINDIR}/libcommon_cuda.so
	nvcc ${CFLAGS} -DPADDING -arch=sm_20 ${SRCDIR}/common.cpp ${SRCDIR}/common3d.cpp -lcommon_cuda $< -o $@
${BINDIR}/iso3d_cuda_mpi2: ${SRCDIR}/iso3d_mpi2.cu ${BINDIR}/libcommon_cuda.so
	nvcc ${CFLAGS} -DPADDING -arch=sm_20 ${SRCDIR}/common.cpp ${SRCDIR}/common3d.cpp -lcommon_cuda $< -o $@
${BINDIR}/iso3d_cuda_mpi: ${SRCDIR}/iso3d_mpi.cu ${BINDIR}/libcommon_cuda.so
	nvcc ${CFLAGS} -DPADDING -arch=sm_20 ${SRCDIR}/common.cpp ${SRCDIR}/common3d.cpp -lcommon_cuda $< -o $@
${BINDIR}/iso3d_cuda_peer: ${SRCDIR}/iso3d_peer.cu ${BINDIR}/libcommon_cuda.so
	nvcc ${CFLAGS} -DPADDING -arch=sm_20 ${SRCDIR}/common.cpp ${SRCDIR}/common3d.cpp -lcommon_cuda $< -o $@

${BINDIR}/iso2d_cuda_peer: ${SRCDIR}/iso2d_peer.cu ${BINDIR}/libcommon_cuda.so
	nvcc ${CFLAGS} -DPADDING -arch=sm_20 ${SRCDIR}/common.cpp ${SRCDIR}/common2d.cpp -lcommon_cuda $< -o $@
${BINDIR}/iso2d_cuda_multigpu: ${SRCDIR}/iso2d_multigpu.cu ${BINDIR}/libcommon_cuda.so
	nvcc ${CFLAGS} -DPADDING -arch=sm_20 ${SRCDIR}/common.cpp ${SRCDIR}/common2d.cpp -lcommon_cuda $< -o $@
${BINDIR}/iso2d_cuda_unsafe: ${SRCDIR}/iso2d_unsafe.cu ${BINDIR}/libcommon_cuda.so
	nvcc ${CFLAGS} -DPADDING -arch=sm_20 ${SRCDIR}/common.cpp ${SRCDIR}/common2d.cpp -lcommon_cuda $< -o $@
${BINDIR}/iso2d_cuda_atomic: ${SRCDIR}/iso2d_atomic.cu ${BINDIR}/libcommon_cuda.so
	nvcc ${CFLAGS} -DPADDING -arch=sm_20 ${SRCDIR}/common.cpp ${SRCDIR}/common2d.cpp -lcommon_cuda $< -o $@
${BINDIR}/iso2d_cuda_coop: ${SRCDIR}/iso2d_coop.cu ${BINDIR}/libcommon_cuda.so
	nvcc ${CFLAGS} -DPADDING -arch=sm_20 ${SRCDIR}/common.cpp ${SRCDIR}/common2d.cpp -lcommon_cuda $< -o $@
${BINDIR}/iso2d_cuda_constant: ${SRCDIR}/iso2d_constant.cu ${BINDIR}/libcommon_cuda.so
	nvcc ${CFLAGS} -DPADDING -arch=sm_20 ${SRCDIR}/common.cpp ${SRCDIR}/common2d.cpp -lcommon_cuda $< -o $@
${BINDIR}/iso2d_cuda_shared: ${SRCDIR}/iso2d_shared.cu ${BINDIR}/libcommon_cuda.so
	nvcc ${CFLAGS} -DPADDING -arch=sm_20 ${SRCDIR}/common.cpp ${SRCDIR}/common2d.cpp -lcommon_cuda $< -o $@
${BINDIR}/iso2d_cuda_global: ${SRCDIR}/iso2d_global.cu ${BINDIR}/libcommon_cuda.so
	nvcc ${CFLAGS} -DPADDING -arch=sm_20 ${SRCDIR}/common.cpp ${SRCDIR}/common2d.cpp -lcommon_cuda $< -o $@
${BINDIR}/iso2d_cuda_unified: ${SRCDIR}/iso2d_unified.cu ${BINDIR}/libcommon_cuda.so
	nvcc ${CFLAGS} -arch=sm_20 ${SRCDIR}/common.cpp ${SRCDIR}/common2d.cpp -lcommon_cuda $< -o $@
${BINDIR}/iso2d_cuda_uva: ${SRCDIR}/iso2d_uva.cu ${BINDIR}/libcommon_cuda.so
	nvcc ${CFLAGS} -arch=sm_20 ${SRCDIR}/common.cpp ${SRCDIR}/common2d.cpp -lcommon_cuda $< -o $@
${BINDIR}/iso2d_cuda_zerocopy: ${SRCDIR}/iso2d_zerocopy.cu ${BINDIR}/libcommon_cuda.so
	nvcc ${CFLAGS} -arch=sm_20 ${SRCDIR}/common.cpp ${SRCDIR}/common2d.cpp -lcommon_cuda $< -o $@
${BINDIR}/iso2d_cuda_pinned: ${SRCDIR}/iso2d_pinned.cu ${BINDIR}/libcommon_cuda.so
	nvcc ${CFLAGS} -arch=sm_20 ${SRCDIR}/common.cpp ${SRCDIR}/common2d.cpp -lcommon_cuda $< -o $@

${BINDIR}/iso2d_cuda: ${SRCDIR}/iso2d.cu ${BINDIR}/libcommon_cuda.so
	nvcc ${CFLAGS} -arch=sm_20 ${SRCDIR}/common.cpp ${SRCDIR}/common2d.cpp -lcommon_cuda $< -o $@
${BINDIR}/iso3d_cuda: ${SRCDIR}/iso3d.cu ${BINDIR}/libcommon_cuda.so
	nvcc ${CFLAGS} -arch=sm_20 ${SRCDIR}/common.cpp ${SRCDIR}/common3d.cpp -lcommon_cuda $< -o $@

${BINDIR}/iso2d_omp: ${SRCDIR}/iso2d.c ${BINDIR}/libcommon_cuda.so
	g++ ${CFLAGS} ${SRCDIR}/common.cpp ${SRCDIR}/common2d.cpp -lcommon_cuda $< -o $@ -fopenmp
${BINDIR}/iso3d_omp: ${SRCDIR}/iso3d.c ${BINDIR}/libcommon_cuda.so
	g++ ${CFLAGS} ${SRCDIR}/common.cpp ${SRCDIR}/common3d.cpp -lcommon_cuda $< -o $@ -fopenmp

${BINDIR}/iso2d: ${SRCDIR}/iso2d.c ${BINDIR}/libcommon_cuda.so
	g++ ${CFLAGS} ${SRCDIR}/common.cpp ${SRCDIR}/common2d.cpp -lcommon_cuda $< -o $@
${BINDIR}/iso3d: ${SRCDIR}/iso3d.c ${BINDIR}/libcommon_cuda.so
	g++ ${CFLAGS} ${SRCDIR}/common.cpp ${SRCDIR}/common3d.cpp -lcommon_cuda $< -o $@

${BINDIR}/libcommon_cuda.so: ${SRCDIR}/common_cuda.cu
	nvcc ${CFLAGS} -arch=sm_20 -shared -Xcompiler -fPIC $< -o $@

clean:
	rm -f ${BINDIR}/*
