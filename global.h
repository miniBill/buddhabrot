#pragma once

int global_width;
int global_height;

double global_minx;
double global_maxx;

double global_miny;
double global_maxy;

double global_centerx;
double global_centery;

double global_fwidth;
double global_fheight;

void global_init(int width, double minx, double maxx, double miny, double maxy) {
    global_minx = minx;
    global_maxx = maxx;

    global_miny = miny;
    global_maxy = maxy;

    global_centerx = (minx + maxx) / 2;
    global_centery = (miny + maxy) / 2;

    global_fwidth = global_maxx - global_minx;
    global_fheight = global_maxy - global_miny;

    global_width = width;
    global_height = width * global_fheight / global_fwidth;
}
