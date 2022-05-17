#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"
#include "color.cuh"
#include "global.h"
#include "ppm.h"

__device__ void go(struct global_t global, double x, double y, uint32_t *hits)
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
      atomicAdd(&hits[y_int * global.width + x_int], 1);

    if (r_squared + i_squared > 4)
      return;
  }
}

__global__ void buddha(
    struct global_t global,
    uint32_t *hits)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int fine_grid = 16;

  double deltax = global.fheight / global.height;
  double deltay = global.fwidth / global.width;

  for (int n = index; n < global.width * global.height; n += stride)
  {
    int x_int = n % global.width;
    int y_int = n / global.width;

    double i = global.miny + y_int * deltax;
    double r = global.minx + x_int * deltay;

    for (int dx = 0; dx < fine_grid; dx++)
      for (int dy = 0; dy < fine_grid; dy++)
        go(global, r + deltax * dx, i + deltay * dy, hits);
  }
}

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    fprintf(stderr, "Usage: %s <output_file.ppm>\n", argc > 0 ? argv[0] : "program");
    return -1;
  }

  struct global_t global = global_init(1920, -2.0, 1.5, -1.5, 1.5);

  uint32_t *hits;
  int N = global.width * global.height;

  // Allocate Unified Memory – accessible from CPU or GPU
  CUDA_CHECK(cudaMallocManaged(&hits, N * sizeof(uint32_t)));

  fprintf(stderr, "Launching kernel\n");

  int blockSize = 512;
  int numBlocks = (N + blockSize - 1) / blockSize;
  buddha<<<numBlocks, blockSize>>>(global, hits);

  fprintf(stderr, "Waiting for kernel to end computation\n");

  // Wait for GPU to finish before accessing on host
  CUDA_CHECK(cudaDeviceSynchronize());

  fprintf(stderr, "Calculating max\n");
  uint32_t max = 0;
  for (int y_int = 0; y_int < global.height; y_int++)
    for (int x_int = 0; x_int < global.width; x_int++)
      if (hits[y_int * global.width + x_int] > max)
        max = hits[y_int * global.width + x_int];
  fprintf(stderr, "Max is %u\n", max);

  char *picture = (char *)malloc(global.width * global.height * 3 * sizeof(char));

  fprintf(stderr, "Calculating colors\n");
  for (int n = 0; n < global.width * global.height; n++)
  {
    int x_int = n % global.width;
    int y_int = n / global.width;
    uint32_t point = hits[n];

    double sqrt_scaled = sqrt(point * 1.0 / max);

    color_t pixel = color_viridis(sqrt_scaled);

    int transposed_n = x_int * global.height + y_int;

    picture[3 * transposed_n] = pixel.r;
    picture[3 * transposed_n + 1] = pixel.g;
    picture[3 * transposed_n + 2] = pixel.b;
  }

  fprintf(stderr, "Outputting the picture - sqrt\n");

  char *filename = argv[1];
  ppm_save_bytes(filename, picture, global.height, global.width);

  // Free memory
  CUDA_CHECK(cudaFree(hits));
  free(picture);

  return 0;
}
