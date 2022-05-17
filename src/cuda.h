#pragma once

#define CUDA_CHECK(call)                                  \
    do                                                    \
    {                                                     \
        cudaError_t status = call;                        \
        if (status != cudaSuccess)                        \
        {                                                 \
            printf("FAIL: call='%s'. Reason:%s\n", #call, \
                   cudaGetErrorString(status));           \
            return -1;                                    \
        }                                                 \
    } while (0)
