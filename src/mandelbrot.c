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

void mbrot(
    int width, int height,
    double fwidth, double fheight,
    double minx, double miny,
    char *result,
    int index, int stride)
{
    for (int n = index; n < width * height; n += stride)
    {
        int x_int = n % width;
        int y_int = n / width;

        double i = miny + y_int * fheight / height;
        double r = minx + x_int * fwidth / width;

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

    global_init(1920 * 4, -2.2, 1.5, -1.2, 1.2);

    char *picture = (char *)calloc(global_width * global_height, 3 * sizeof(char));

    mbrot(
        global_width, global_height,
        global_fwidth, global_fheight,
        global_minx, global_miny,
        picture, 0, 1);

    ppm_save_bytes(filename, picture, global_width, global_height);

    return 0;
}
