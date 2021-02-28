#pragma once

#include <stdint.h>

struct color_t {
    uint8_t r, g, b;
};

struct vec3_t {
    double x,y,z;
};

struct vec3_t plus(struct vec3_t l, struct vec3_t r) {
    struct vec3_t result = { l.x + r.x, l.y + r.y, l.z + r.z };
    return result;
}

struct vec3_t by(double l, struct vec3_t r) {
    struct vec3_t result = { l * r.x, l * r.y, l * r.z };
    return result;
}

uint8_t toInt(double val) {
    if(val < 0) return 0;
    if(val >= 1) return 255;
    return (uint8_t)(256 * val);
}

struct color_t viridis(double t) {
    // https://www.shadertoy.com/view/WlfXRN
    struct vec3_t c0 = { 0.2777273272234177, 0.005407344544966578, 0.3340998053353061 };
    struct vec3_t c1 = { 0.1050930431085774, 1.404613529898575, 1.384590162594685 };
    struct vec3_t c2 = { -0.3308618287255563, 0.214847559468213, 0.09509516302823659 };
    struct vec3_t c3 = { -4.634230498983486, -5.799100973351585, -19.33244095627987 };
    struct vec3_t c4 = { 6.228269936347081, 14.17993336680509, 56.69055260068105 };
    struct vec3_t c5 = { 4.776384997670288, -13.74514537774601, -65.35303263337234 };
    struct vec3_t c6 = { -5.435455855934631, 4.645852612178535, 26.3124352495832 };

    struct vec3_t result_float = plus(c0, by(t, (plus(c1, by(t, (plus(c2, by(t, (plus(c3, by(t, (plus(c4, by(t, (plus(c5, by(t, c6)))))))))))))))));
    struct color_t result = { toInt(result_float.x), toInt(result_float.y), toInt(result_float.z) };
    return result;
}
