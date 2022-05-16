#pragma once

#include "vec3.h"

__device__ struct vec3_t vec3_plus_vec3_cuda(struct vec3_t l, struct vec3_t r)
{
    struct vec3_t result = {l.x + r.x, l.y + r.y, l.z + r.z};
    return result;
}

__device__ struct vec3_t vec3_by_double_cuda(double l, struct vec3_t r)
{
    struct vec3_t result = {l * r.x, l * r.y, l * r.z};
    return result;
}
