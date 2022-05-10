#include <math.h>
#include <stdlib.h>

#include "color.h"
#include "global.h"
#include "ppm.h"

struct color_t go(double x, double y) {
    struct color_t result = { 0, 0, 0 };

    double r = x;
    double i = y;
    double r_squared = r * r;
    double i_squared = i * i;

    for(int c = 0; c < 200; c++)
    {
        if(r_squared + i_squared > 4)
        {
            return color_viridis(log(1 + c / 200.0));
        }

        i = 2 * r * i + y;
        r = r_squared - i_squared + x;

        r_squared = r * r;
        i_squared = i * i;
    }
    return result;
}

int main (int argc, char * argv[]) {
    if(argc < 2) {
        fprintf(stderr, "Usage: %s <output_file.ppm>\n", argc > 0 ? argv[0] : "mandelbrot");
        return -1;
    }

    char * filename = argv[1];

    global_init(1920, -2.2, 1.5, -1.2, 1.2);

    struct color_t * picture = (struct color_t *)calloc(global_width * global_height, sizeof(struct color_t));

    for(int y_int = 0; y_int < global_height; y_int++) {
        double y = global_miny + y_int * global_fheight / global_height;
        for(int x_int = 0; x_int < global_width; x_int++) {
            double x = global_minx + x_int * global_fwidth / global_width;

            picture[y_int * global_width + x_int] = go(x, y);
        }
    }

    ppm_save(filename, picture, global_width, global_height);

    return 0;
}
