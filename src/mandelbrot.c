#include <math.h>
#include <stdlib.h>

#include "color.h"
#include "global.h"
#include "ppm.h"

struct color_t go(double x, double y)
{
    struct color_t result = {0, 0, 0};

    double r = x;
    double i = y;
    double r_squared = r * r;
    double i_squared = i * i;

    for (int c = 0; c < 200; c++)
    {
        if (r_squared + i_squared > 4)
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

void mbrot(struct global_t global, char *result, int index, int stride)
{
    for (int n = index; n < global.width * global.height; n += stride)
    {
        int x_int = n % global.width;
        int y_int = n / global.width;

        double i = global.miny + y_int * global.fheight / global.height;
        double r = global.minx + x_int * global.fwidth / global.width;

        struct color_t point = go(r, i);
        result[3 * n] = point.r;
        result[3 * n + 1] = point.g;
        result[3 * n + 2] = point.b;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <output_file.ppm>\n", argc > 0 ? argv[0] : "mandelbrot");
        return -1;
    }

    char *filename = argv[1];

    struct global_t global = global_init(1920 * 4, -2.2, 1.5, -1.2, 1.2);

    char *picture = (char *)calloc(global.width * global.height, 3 * sizeof(char));

    mbrot(global, picture, 0, 1);

    ppm_save_bytes(filename, picture, global.width, global.height);

    return 0;
}
