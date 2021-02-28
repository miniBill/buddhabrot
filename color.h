#pragma once

#include <stdint.h>

#include "vec3.h"

struct color_t {
    uint8_t r, g, b;
};

uint8_t color_double_to_int(double val) {
    if(val < 0) return 0;
    if(val >= 1) return 255;
    return (uint8_t)(256 * val);
}

struct color_t color_viridis(double t) {
    // https://www.shadertoy.com/view/WlfXRN
    struct vec3_t c0 = { 0.2777273272234177, 0.005407344544966578, 0.3340998053353061 };
    struct vec3_t c1 = { 0.1050930431085774, 1.404613529898575, 1.384590162594685 };
    struct vec3_t c2 = { -0.3308618287255563, 0.214847559468213, 0.09509516302823659 };
    struct vec3_t c3 = { -4.634230498983486, -5.799100973351585, -19.33244095627987 };
    struct vec3_t c4 = { 6.228269936347081, 14.17993336680509, 56.69055260068105 };
    struct vec3_t c5 = { 4.776384997670288, -13.74514537774601, -65.35303263337234 };
    struct vec3_t c6 = { -5.435455855934631, 4.645852612178535, 26.3124352495832 };

    struct vec3_t result_float = vec3_plus_vec3(c0, vec3_by_double(t, (vec3_plus_vec3(c1, vec3_by_double(t, (vec3_plus_vec3(c2, vec3_by_double(t, (vec3_plus_vec3(c3, vec3_by_double(t, (vec3_plus_vec3(c4, vec3_by_double(t, (vec3_plus_vec3(c5, vec3_by_double(t, c6)))))))))))))))));
    struct color_t result = { color_double_to_int(result_float.x), color_double_to_int(result_float.y), color_double_to_int(result_float.z) };
    return result;
}
