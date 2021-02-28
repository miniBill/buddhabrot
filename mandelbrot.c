#include <math.h>
#include <stdlib.h>

#include "color.h"
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
            return viridis(log(1 + c / 200.0));
        }

        i = 2 * r * i + y;
        r = r_squared - i_squared + x;

        r_squared = r * r;
        i_squared = i * i;
    }
    return result;
}

int main () {
    int width = 1920;

    int height = width * 20 / 39;

    double minx = -2.2;
    double maxx = 1.5;

    double fwidth = maxx - minx;
    double fheight = fwidth * height / width;

    double centery = 0;

    double miny = centery - fheight / 2;

    struct color_t * picture = calloc(width * height, sizeof(struct color_t));

    for(int y_int = 0; y_int < height; y_int++) {
        double y = miny + y_int * fheight / height;
        for(int x_int = 0; x_int < width; x_int++) {
            double x = minx + x_int * fwidth / width;

            picture[y_int * width + x_int] = go(x, y);
        }
    }

    output_as_ppm(picture, width, height);

    free(picture);

    return 0;
}
