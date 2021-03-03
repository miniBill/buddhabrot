#pragma once

#include <stdio.h>

#include "color.h"

void ppm_save(char * filename, struct color_t * picture, int width, int height) {
    FILE * f = fopen(filename, "w+");
    fprintf(f, "P6\n");
    fprintf(f, "%d %d\n255\n", width, height);

    for(int y = height - 1; y>=0; y--) {
        for(int x = 0; x < width; x++) {
            struct color_t color = picture[y * width + x];
            fprintf(f, "%c%c%c", (char)color.r, (char)color.g, (char)color.b);
        }
    }
    fclose(f);
}
