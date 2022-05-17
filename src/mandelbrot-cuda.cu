#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"
#include "color.cuh"
#include "global.h"
#include "ppm.h"

__device__ struct color_t go(double x, double y)
{
  double r = x;
  double i = y;
  double r_squared = r * r;
  double i_squared = i * i;

  for (int c = 0; c < 200; c++)
  {
    if (r_squared + i_squared > 4)
    {
      return color_viridis_cuda(log(1 + c / 200.0));
    }

    i = 2 * r * i + y;
    r = r_squared - i_squared + x;

    r_squared = r * r;
    i_squared = i * i;
  }

  return {0, 0, 0};
}

__global__ void mbrot(
    struct global_t global,
    char *result)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int n = index; n < global.width * global.height; n += stride)
  {
    int x_int = n % global.width;
    int y_int = n / global.width;

    double i = global.miny + y_int * global.fheight / global.height;
    double r = global.minx + x_int * global.fwidth / global.width;

    color_t point = go(r, i);
    result[3 * n] = point.r;
    result[3 * n + 1] = point.g;
    result[3 * n + 2] = point.b;
  }
}

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    fprintf(stderr, "Usage: %s <output_file.ppm>\n", argc > 0 ? argv[0] : "program");
    return -1;
  }

  global_t global = global_init(1920 * 4, -2.2, 1.5, -1.2, 1.2);

  char *picture;
  int N = global.width * global.height;

  // Allocate Unified Memory – accessible from CPU or GPU
  CUDA_CHECK(cudaMallocManaged(&picture, N * 3));

  int blockSize = 512;
  int numBlocks = (N + blockSize - 1) / blockSize;
  mbrot<<<numBlocks, blockSize>>>(global, picture);

  // Wait for GPU to finish before accessing on host
  CUDA_CHECK(cudaDeviceSynchronize());

  char *filename = argv[1];
  ppm_save_bytes(filename, picture, global.width, global.height);

  // Free memory
  CUDA_CHECK(cudaFree(picture));

  return 0;
}
