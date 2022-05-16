#pragma once

struct global_t
{
    int width;
    int height;

    double minx;
    double maxx;

    double miny;
    double maxy;

    double centerx;
    double centery;

    double fwidth;
    double fheight;
};

struct global_t global_init(int width, double minx, double maxx, double miny, double maxy)
{
    struct global_t result;
    result.minx = minx;
    result.maxx = maxx;

    result.miny = miny;
    result.maxy = maxy;

    result.centerx = (minx + maxx) / 2;
    result.centery = (miny + maxy) / 2;

    result.fwidth = result.maxx - result.minx;
    result.fheight = result.maxy - result.miny;

    result.width = width;
    result.height = width * result.fheight / result.fwidth;
    return result;
}
