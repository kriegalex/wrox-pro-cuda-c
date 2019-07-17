#include <openacc.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv) {
    acc_device_t type = acc_device_nvidia;
    const char *type_name = "acc_device_nvidia";

    int naccelerators = acc_get_num_devices(type);
    printf("This OpenACC installation has %d accelerators of type %s(%d)\n",
        naccelerators, type_name, type);

    for (int i = 0; i < naccelerators; i++) {
        acc_set_device_num(i, type);
        printf("Switched to accelerator %d, device type=%d\n",
            acc_get_device_num(type), acc_get_device_type());
    }

    acc_set_device_type(type);
    printf("Using acc_set_device_type put us on accelerator %d, type %d\n",
        acc_get_device_num(type), acc_get_device_type());

    return 0;
}
