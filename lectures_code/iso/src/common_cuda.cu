#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

int getNumCUDADevices() {
    int ndevices;
    CHECK(cudaGetDeviceCount(&ndevices));
    return ndevices;
}

#ifdef __cplusplus
}
#endif
