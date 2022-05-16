#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "global.h"
#include "ppm.h"

#define CUDA_CHECK(call)                            \
  do                                                \
  {                                                 \
    cudaError_t status = call;                      \
    if (status != cudaSuccess)                      \
    {                                               \
      printf("FAIL: call='%s'. Reason:%s\n", #call, \
             cudaGetErrorString(status));           \
      return -1;                                    \
    }                                               \
  } while (0)

__device__ uint8_t color_double_to_int_cuda(double val)
{
  if (val < 0)
    return 0;
  if (val >= 1)
    return 255;
  return (uint8_t)(256 * val);
}

__device__ struct vec3_t vec3_plus_vec3_cuda(struct vec3_t l, struct vec3_t r)
{
  struct vec3_t result = {l.x + r.x, l.y + r.y, l.z + r.z};
  return result;
}

__device__ struct vec3_t vec3_by_double_cuda(double l, struct vec3_t r)
{
  struct vec3_t result = {l * r.x, l * r.y, l * r.z};
  return result;
}

__device__ struct color_t color_viridis_cuda(double t)
{
  // https://www.shadertoy.com/view/WlfXRN
  struct vec3_t c0 = {0.2777273272234177, 0.005407344544966578, 0.3340998053353061};
  struct vec3_t c1 = {0.1050930431085774, 1.404613529898575, 1.384590162594685};
  struct vec3_t c2 = {-0.3308618287255563, 0.214847559468213, 0.09509516302823659};
  struct vec3_t c3 = {-4.634230498983486, -5.799100973351585, -19.33244095627987};
  struct vec3_t c4 = {6.228269936347081, 14.17993336680509, 56.69055260068105};
  struct vec3_t c5 = {4.776384997670288, -13.74514537774601, -65.35303263337234};
  struct vec3_t c6 = {-5.435455855934631, 4.645852612178535, 26.3124352495832};

  struct vec3_t result_float = vec3_plus_vec3_cuda(c0, vec3_by_double_cuda(t, vec3_plus_vec3_cuda(c1, vec3_by_double_cuda(t, vec3_plus_vec3_cuda(c2, vec3_by_double_cuda(t, vec3_plus_vec3_cuda(c3, vec3_by_double_cuda(t, vec3_plus_vec3_cuda(c4, vec3_by_double_cuda(t, vec3_plus_vec3_cuda(c5, vec3_by_double_cuda(t, c6))))))))))));
  struct color_t result = {color_double_to_int_cuda(result_float.x), color_double_to_int_cuda(result_float.y), color_double_to_int_cuda(result_float.z)};
  return result;
}

__device__ struct color_t go_cuda(double x, double y)
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
    int width, int height,
    double fwidth, double fheight,
    double minx, double miny,
    char *result)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int n = index; n < width * height; n += stride)
  {
    int x_int = n % width;
    int y_int = n / width;

    double i = miny + y_int * fheight / height;
    double r = minx + x_int * fwidth / width;

    color_t point = go_cuda(r, i);
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

  global_init(1920 * 4, -2.2, 1.5, -1.2, 1.2);

  char *picture;
  int N = global_width * global_height;

  // Allocate Unified Memory – accessible from CPU or GPU
  CUDA_CHECK(cudaMallocManaged(&picture, N * 3));

  // int devId;
  // CUDA_CHECK(cudaGetDevice(&devId));

  // int smemSize;
  // CUDA_CHECK(cudaDeviceGetAttribute(&smemSize, cudaDevAttrMaxSharedMemoryPerBlock, devId));
  // int numProcs;
  // CUDA_CHECK(cudaDeviceGetAttribute(&numProcs, cudaDevAttrMultiProcessorCount, devId));

  // printf("Shared memory per block: %d, MP count %d\n", smemSize, numProcs);

  // Run kernel on 1M elements on the GPU
  int blockSize = 512;
  int numBlocks = (N + blockSize - 1) / blockSize;
  mbrot<<<numBlocks, blockSize>>>(
      global_width, global_height,
      global_fwidth, global_fheight,
      global_minx, global_miny,
      picture);

  // Wait for GPU to finish before accessing on host
  CUDA_CHECK(cudaDeviceSynchronize());

  char *filename = argv[1];
  ppm_save_bytes(filename, picture, global_width, global_height);

  // Free memory
  CUDA_CHECK(cudaFree(picture));

  return 0;
}
