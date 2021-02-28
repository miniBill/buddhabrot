#include <math.h>
#include <stdlib.h>

#include "color.h"
#include "global.h"
#include "ppm.h"

void go(double x, double y, uint64_t * hits) {
    double r = x;
    double i = y;
    double r_squared = r * r;
    double i_squared = i * i;

    int c;

    int max_iterations = 600;

    for(c = 0; c < max_iterations; c++)
    {
        i = 2 * r * i + y;
        r = r_squared - i_squared + x;

        r_squared = r * r;
        i_squared = i * i;
        
        if(r_squared + i_squared > 4)
            break;
    }

    if(c == max_iterations)
        return;

    r = x;
    i = y;
    r_squared = r * r;
    i_squared = i * i;

    for(int c = 0; c < max_iterations; c++)
    {
        i = 2 * r * i + y;
        r = r_squared - i_squared + x;

        r_squared = r * r;
        i_squared = i * i;

        int x_int = (r - global_minx) / global_fwidth * global_width;
        int y_int = (i - global_miny) / global_fheight * global_height;

        if(x_int >= 0 && x_int < global_width && y_int >= 0 && y_int < global_height)
            hits[y_int * global_width + x_int]++;

        if(r_squared + i_squared > 4)
            return;
    }
}

int main () {
    global_init(1920, -2.0, 1.5, -1.5, 1.5);

    struct color_t * picture = (struct color_t *)calloc(global_width * global_height, sizeof(struct color_t));
    uint64_t * hits = (uint64_t *)calloc(global_width * global_height, sizeof(uint64_t));

    int multiplier = 4;
    for(int y_int = 0; y_int < global_height * multiplier; y_int++) {
        double y = global_miny + y_int * global_fheight / global_height / multiplier;
        for(int x_int = 0; x_int < global_width * multiplier; x_int++) {
            double x = global_minx + x_int * global_fwidth / global_width / multiplier;
            go(x, y, hits);
        }
    }

    uint64_t max = 0;
    for(int y_int = 0; y_int < global_height; y_int++)
        for(int x_int = 0; x_int < global_width; x_int++)
            if(hits[y_int * global_width + x_int] > max)
                max = hits[y_int * global_width + x_int];

    for(int y_int = 0; y_int < global_height; y_int++) {
        for(int x_int = 0; x_int < global_width; x_int++) {
            uint64_t point = hits[y_int * global_width + x_int];
            double scaled = sqrt(point * 1.0) / sqrt(max * 1.0);
            picture[(global_width - 1 - x_int) * global_height + y_int ] = color_viridis(scaled);
        }
    }

    ppm_save(picture, global_height, global_width);

    free(picture);

    return 0;
}
