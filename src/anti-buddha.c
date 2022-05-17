#include <math.h>
#include <stdlib.h>

#include "color.h"
#include "global.h"
#include "ppm.h"

void go(struct global_t global, double x, double y, uint64_t *hits)
{
    double r = x;
    double i = y;
    double r_squared = r * r;
    double i_squared = i * i;

    int c;

    int max_iterations = 600;

    for (c = 0; c < max_iterations; c++)
    {
        i = 2 * r * i + y;
        r = r_squared - i_squared + x;

        r_squared = r * r;
        i_squared = i * i;

        if (r_squared + i_squared > 4)
            return;
    }

    r = x;
    i = y;
    r_squared = r * r;
    i_squared = i * i;

    for (int c = 0; c < max_iterations; c++)
    {
        i = 2 * r * i + y;
        r = r_squared - i_squared + x;

        r_squared = r * r;
        i_squared = i * i;

        int x_int = (r - global.minx) / global.fwidth * global.width;
        int y_int = (i - global.miny) / global.fheight * global.height;

        if (x_int >= 0 && x_int < global.width && y_int >= 0 && y_int < global.height)
            hits[y_int * global.width + x_int]++;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <output_file.ppm>\n", argc > 0 ? argv[0] : "anti-buddha");
        return -1;
    }

    char *filename = argv[1];

    struct global_t global = global_init(1920, -2.0, 1.5, -1.5, 1.5);

    struct color_t *picture = (struct color_t *)calloc(global.width * global.height, sizeof(struct color_t));
    uint64_t *hits = (uint64_t *)calloc(global.width * global.height, sizeof(uint64_t));

    int multiplier = 4;
    for (int y_int = 0; y_int < global.height * multiplier; y_int++)
    {
        double y = global.miny + y_int * global.fheight / global.height / multiplier;
        for (int x_int = 0; x_int < global.width * multiplier; x_int++)
        {
            double x = global.minx + x_int * global.fwidth / global.width / multiplier;
            go(global, x, y, hits);
        }
    }

    uint64_t max = 0;
    for (int y_int = 0; y_int < global.height; y_int++)
        for (int x_int = 0; x_int < global.width; x_int++)
            if (hits[y_int * global.width + x_int] > max)
                max = hits[y_int * global.width + x_int];

    for (int y_int = 0; y_int < global.height; y_int++)
    {
        for (int x_int = 0; x_int < global.width; x_int++)
        {
            uint64_t point = hits[y_int * global.width + x_int];
            double scaled = sqrt(point * 1.0) / sqrt(max * 1.0);
            picture[(global.width - 1 - x_int) * global.height + y_int] = color_viridis(scaled);
        }
    }

    ppm_save(filename, picture, global.height, global.width);

    return 0;
}
