#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysinfo.h>
#include <threads.h>

#include "color.h"
#include "global.h"
#include "ppm.h"

void go(struct global_t global, double x, double y, uint32_t *hits)
{
    double r = x;
    double i = y;
    double r_squared = x * x;
    double i_squared = y * y;
    double i_old = 0;
    double r_old = 0;

    double p = sqrt((x - 0.25) * (x - 0.25) + i_squared);
    if (x <= p - 2 * p * p + 0.25 || (x + 1) * (x + 1) + i_squared <= 1.0 / 16)
        return;

    int max_iterations = 16 * 1000 * 1000;

    int c;
    for (c = 1; r_squared + i_squared < 4 && c < max_iterations; c++)
    {
        i = 2 * r * i + y;
        r = r_squared - i_squared + x;

        r_squared = r * r;
        i_squared = i * i;

        if (i == i_old && r == r_old)
            return;

        if ((c & (c - 1)) == 0)
        {
            i_old = i;
            r_old = r;
        }
    }

    if (c == max_iterations)
        return;

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

        if (r_squared + i_squared > 4)
            return;
    }
}

struct slice_init_t
{
    struct global_t global;
    int skip;
};

void *slice(void *void_arg)
{
    struct slice_init_t init = *(struct slice_init_t *)void_arg;
    int skip = init.skip;
    struct global_t global = init.global;
    uint32_t *hits = (uint32_t *)calloc(global.width * global.height, sizeof(uint32_t));

    int thread_count = get_nprocs();

    int finer_grid_multiplier = 16;

    // int old_percentage = -1;

    for (int y_int = skip; y_int < global.height * finer_grid_multiplier / 2; y_int += thread_count)
    {
        // int percentage = y_int * 100 / global.height / finer_grid_multiplier;
        // if (skip == thread_count - 1 && percentage != old_percentage)
        // {
        //     fprintf(stderr, "%d%% ", percentage);
        //     if (percentage == 69)
        //         fprintf(stderr, "(nice!) ");
        //     fflush(stderr);
        //     old_percentage = percentage;
        // }
        double y = global.miny + y_int * global.fheight / global.height / finer_grid_multiplier;
        for (int x_int = 0; x_int < global.width * finer_grid_multiplier; x_int++)
        {
            double x = global.minx + x_int * global.fwidth / global.width / finer_grid_multiplier;
            go(global, x, y, hits);
        }
    }

    if (skip == thread_count - 1)
        fprintf(stderr, "100%%\n");

    return hits;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <output_file.ppm>\n", argc > 0 ? argv[0] : "buddha");
        return -1;
    }

    char *filename = argv[1];

    struct global_t global = global_init(1920, -2.0, 1.5, -1.5, 1.5);

    struct color_t *picture = (struct color_t *)calloc(global.width * global.height, sizeof(struct color_t));
    uint32_t *hits = (uint32_t *)calloc(global.width * global.height, sizeof(uint32_t));

    int thread_count = get_nprocs();

    pthread_t threads[thread_count];
    struct slice_init_t skips[thread_count];

    for (int i = 0; i < thread_count; i++)
    {
        skips[i] = (struct slice_init_t){global, i};
        fprintf(stderr, "Starting thread #%d\n", i + 1);
        if (pthread_create(&threads[i], NULL, slice, &skips[i]))
        {
            fprintf(stderr, "Error creating thread #%d\n", i + 1);
            return 1;
        }
    }

    for (int i = thread_count - 1; i >= 0; i--)
    {
        uint32_t *thread_hits;

        fprintf(stderr, "Waiting thread #%d\n", i + 1);
        if (pthread_join(threads[i], (void *)(&thread_hits)))
        {
            fprintf(stderr, "Error joining thread #%d\n", i + i);
            return 2;
        }

        for (int y_int = 0; y_int < global.height; y_int++)
            for (int x_int = 0; x_int < global.width; x_int++)
                hits[y_int * global.width + x_int] += thread_hits[y_int * global.width + x_int] + thread_hits[(global.height - y_int - 1) * global.width + x_int];
    }

    fprintf(stderr, "Calculating max\n");
    uint32_t max = 0;
    for (int y_int = 0; y_int < global.height; y_int++)
        for (int x_int = 0; x_int < global.width; x_int++)
            if (hits[y_int * global.width + x_int] > max)
                max = hits[y_int * global.width + x_int];
    fprintf(stderr, "Max is %" PRIu32 "\n", max);

    fprintf(stderr, "Calculating colors\n");
    for (int y_int = 0; y_int < global.height; y_int++)
    {
        for (int x_int = 0; x_int < global.width; x_int++)
        {
            uint32_t point = hits[y_int * global.width + x_int];

            double sqrt_scaled = sqrt(point * 1.0 / max);

            int addr = (global.width - 1 - x_int) * global.height + y_int;
            picture[addr] = color_viridis(sqrt_scaled);
        }
    }

    fprintf(stderr, "Outputting the picture - sqrt\n");
    ppm_save(filename, picture, global.height, global.width);

    return 0;
}
