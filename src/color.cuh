#pragma once

#include "color.h"
#include "vec3.cuh"

__device__ uint8_t color_double_to_int_cuda(double val)
{
    if (val < 0)
        return 0;
    if (val >= 1)
        return 255;
    return (uint8_t)(256 * val);
}

__device__ struct color_t color_viridis_cuda(double t)
{
    // https://www.shadertoy.com/view/WlfXRN
    struct vec3_t c0 = {0.2777273272234177, 0.005407344544966578, 0.3340998053353061};
    struct vec3_t c1 = {0.1050930431085774, 1.404613529898575, 1.384590162594685};
    struct vec3_t c2 = {-0.3308618287255563, 0.214847559468213, 0.09509516302823659};
    struct vec3_t c3 = {-4.634230498983486, -5.799100973351585, -19.33244095627987};
    struct vec3_t c4 = {6.228269936347081, 14.17993336680509, 56.69055260068105};
    struct vec3_t c5 = {4.776384997670288, -13.74514537774601, -65.35303263337234};
    struct vec3_t c6 = {-5.435455855934631, 4.645852612178535, 26.3124352495832};

    struct vec3_t result_float = vec3_plus_vec3_cuda(c0, vec3_by_double_cuda(t, vec3_plus_vec3_cuda(c1, vec3_by_double_cuda(t, vec3_plus_vec3_cuda(c2, vec3_by_double_cuda(t, vec3_plus_vec3_cuda(c3, vec3_by_double_cuda(t, vec3_plus_vec3_cuda(c4, vec3_by_double_cuda(t, vec3_plus_vec3_cuda(c5, vec3_by_double_cuda(t, c6))))))))))));
    struct color_t result = {color_double_to_int_cuda(result_float.x), color_double_to_int_cuda(result_float.y), color_double_to_int_cuda(result_float.z)};
    return result;
}
