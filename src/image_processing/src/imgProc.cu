///-----------------------------------------------------------------------------
/// @file imgProc.cu
///
/// @author Riley Kenyon (rike2277@colorado.edu)
/// @brief
///
/// @date 2024-01-27
///-----------------------------------------------------------------------------
#include "imgProc.hpp"
// #include <cuda.h>
// #include <cuda_runtime_api.h>

void convert_grayscale_cpu(cv::Mat *img, cv::Mat *grayscale) {
  for (int i = 0; i < img->rows * img->cols; i++) {
    grayscale[i] = img->data[3 * i] * 0.07 + img->data[3 * i + 1] * 0.72 + img->data[3 * i + 2] * 0.21;
  }
}

__global__ void gpu_grayscale(const unsigned char *mat, unsigned char *matG, int width, int height) {
  // Distance between array elements (i,j)[0] to (i,j)[1] is 1 not width*height
  int tid = blockIdx.x * blockDim.x + threadIdx.x; // thread ID
  int stride = blockDim.x * gridDim.x;             // stride lengths

  // grayscale calculation with strides
  while (tid < width * height) {
    matG[tid] = mat[3 * tid] * 0.07 + mat[3 * tid + 1] * 0.72 + mat[3 * tid + 2] * 0.21;
    // if (matG[tid] > 170) { // saturate to either 255 or 0 - for pixel testing
    //   matG[tid] = 255;
    // } else {
    //   matG[tid] = 0;
    // }
    tid = tid + stride;
  }
}