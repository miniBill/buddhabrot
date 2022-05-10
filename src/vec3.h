#pragma once

struct vec3_t {
    double x,y,z;
};

struct vec3_t vec3_plus_vec3(struct vec3_t l, struct vec3_t r) {
    struct vec3_t result = { l.x + r.x, l.y + r.y, l.z + r.z };
    return result;
}

struct vec3_t vec3_by_double(double l, struct vec3_t r) {
    struct vec3_t result = { l * r.x, l * r.y, l * r.z };
    return result;
}
