#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysinfo.h>
#include <threads.h>

#include "color.h"
#include "global.h"
#include "ppm.h"

void go(double x, double y, uint64_t * hits) {
    double r = x;
    double i = y;
    double r_squared = x * x;
    double i_squared = y * y;
    double i_old = 0;
    double r_old = 0;

    double p = sqrt((x-0.25)*(x-0.25) + i_squared);
    if(x <= p - 2*p*p + 0.25 || (x+1)*(x+1) + i_squared <= 1.0 / 16)
        return;

    int max_iterations = 1000 * 1000;

    int c;
    for(c = 1; r_squared + i_squared < 4 && c < max_iterations; c++)
    {
        i = 2 * r * i + y;
        r = r_squared - i_squared + x;

        r_squared = r * r;
        i_squared = i * i;

        if(i == i_old && r == r_old)
            return;

        if((c & (c - 1)) == 0) {
            i_old = i;
            r_old = r;
        }
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

void * slice(void * skip_void) {
    int skip = *(int*)skip_void;
    uint64_t * hits = (uint64_t *)calloc(global_width * global_height, sizeof(uint64_t));

    int thread_count = get_nprocs();

    int finer_grid_multiplier = 4;
    int coarser_grid_multiplier = 1;

    int old_percentage = -1;

    for(int y_int = skip; y_int < global_height * finer_grid_multiplier / coarser_grid_multiplier / 2; y_int += thread_count) {
        int percentage = y_int * 100 * coarser_grid_multiplier / global_height / finer_grid_multiplier;
        if(skip == thread_count - 1 && percentage != old_percentage) {
            fprintf(stderr, "%d%% ", percentage);
            if(percentage == 69)
                fprintf(stderr, "(nice!) ");
            fflush(stderr);
            old_percentage = percentage;
        }
        double y = global_miny + y_int * global_fheight / global_height / finer_grid_multiplier * coarser_grid_multiplier;
        for(int x_int = 0; x_int < global_width * finer_grid_multiplier / coarser_grid_multiplier; x_int++) {
            double x = global_minx + x_int * global_fwidth / global_width / finer_grid_multiplier * coarser_grid_multiplier;
            go(x, y, hits);
        }
    }

    if(skip == thread_count - 1)
        fprintf(stderr, "100%%\n");

    return hits;
}

int main (int argc, char * argv[]) {
    if(argc < 2) {
        fprintf(stderr, "Usage: %s <output_file.ppm>\n", argc > 0 ? argv[0] : "buddha");
        return -1;
    }

    char * filename = argv[1];

    char * sqrt_filename = calloc(strlen(filename) + strlen("-sqrt") + 1, sizeof(char));
    strcpy(sqrt_filename, filename);
    strcpy(sqrt_filename + strlen(filename), "-sqrt");

    char * linear_filename = calloc(strlen(filename) + strlen("-linear") + 1, sizeof(char));
    strcpy(linear_filename, filename);
    strcpy(linear_filename + strlen(filename), "-linear");

    global_init(1920 * 8, -2.0, 1.5, -1.5, 1.5);

    struct color_t * sqrt_picture = (struct color_t *)calloc(global_width * global_height, sizeof(struct color_t));
    struct color_t * linear_picture = (struct color_t *)calloc(global_width * global_height, sizeof(struct color_t));
    uint64_t * hits = (uint64_t *)calloc(global_width * global_height, sizeof(uint64_t));

    int thread_count = get_nprocs();

    pthread_t threads[thread_count];
    int skips[thread_count];

    for(int i = 0; i < thread_count; i++) {
        skips[i] = i;
        fprintf(stderr, "Starting thread #%d\n", i + 1);
        if(pthread_create(&threads[i], NULL, slice, &skips[i])) {
            fprintf(stderr, "Error creating thread #%d\n", i + 1);
            return 1;
        }
    }

    for(int i = thread_count - 1; i >= 0 ; i--) {
        uint64_t * thread_hits;

        fprintf(stderr, "Waiting thread #%d\n", i + 1);
        if(pthread_join(threads[i], (void*)(&thread_hits))) {
            fprintf(stderr, "Error joining thread #%d\n", i + i);
            return 2;
        }

        for(int y_int = 0; y_int < global_height; y_int++)
            for(int x_int = 0; x_int < global_width; x_int++)
                hits[y_int * global_width + x_int] += thread_hits[y_int * global_width + x_int];
    }

    fprintf(stderr, "Calculating max\n");
    uint64_t max = 0;
    for(int y_int = 0; y_int < global_height; y_int++)
        for(int x_int = 0; x_int < global_width; x_int++)
            if(hits[y_int * global_width + x_int] > max)
                max = hits[y_int * global_width + x_int];
    fprintf(stderr, "Max is %llf\n", max);

    fprintf(stderr, "Calculating colors\n");
    for(int y_int = 0; y_int < global_height; y_int++) {
        for(int x_int = 0; x_int < global_width; x_int++) {
            uint64_t point = hits[y_int * global_width + x_int];

            double linear_scaled = point * 1.0 / max;
            double sqrt_scaled = sqrt(linear_scaled);

            int addr = (global_width - 1 - x_int) * global_height + y_int;
            linear_picture[addr] = color_viridis(linear_scaled);
            sqrt_picture[addr] = color_viridis(sqrt_scaled);
        }
    }

    fprintf(stderr, "Outputting the picture - linear\n");
    ppm_save(linear_filename, linear_picture, global_height, global_width);

    fprintf(stderr, "Outputting the picture - sqrt\n");
    ppm_save(sqrt_filename, sqrt_picture, global_height, global_width);

    return 0;
}
