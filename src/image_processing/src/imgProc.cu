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
    tid = tid + stride;
  }
}

__global__ void screenAllocate(const unsigned char *originalImage,
                               unsigned char *screenImage,
                               const int *imageInfo,
                               const int *screenInfo) {
  // Distance between array elements (i,j)[0] to (i,j)[1] is 1 not width*height
  // thread ID
  int imageWidth = imageInfo[0];
  int imageHeight = imageInfo[1];
  int screenX = screenInfo[0];
  int screenY = screenInfo[1];
  int screenWidth = screenInfo[2];
  int screenHeight = screenInfo[3];
  int index = 0;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  while (tid < screenWidth * screenHeight) {
    index = imageWidth * (screenY + tid / screenWidth) + screenX + (tid - screenWidth * (tid / screenWidth));
    screenImage[tid] = originalImage[index];
    tid = tid + stride;
  }
}

__global__ void edgeFind(const unsigned char *grayData,
                         unsigned char *edge,
                         int width,
                         int height,
                         int threshold = 140) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  while (tid < width * height) {
    if (tid > 3 * width) {
      if (grayData[tid] > threshold && grayData[tid - width] < threshold && grayData[tid - width - 1] < threshold &&
          grayData[tid - width + 1] < threshold && grayData[tid - 1] > threshold) { // probably easier way to do this
        edge[tid] = 255;                                                            // set to white
      } else {
        edge[tid] = 0;
      }
    }
    tid = tid + stride;
  }
}

__global__ void addArr(const unsigned char *arrA,
                       const unsigned char *arrB,
                       unsigned char *output,
                       int width,
                       int height) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  while (tid < width * height) {
    output[tid] = arrA[tid] + arrB[tid];
    tid = tid + stride;
  }
}

__global__ void spacing(const unsigned char *pixelData,
                        unsigned int *difference,
                        unsigned int *count,
                        unsigned int width,
                        int height) {
  // extern__shared__int difference[];
  // extern__shared__int count;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  if (tid < width * height) {
    if (pixelData[tid] == 255) {
      int init = tid;
      int i = init + 10 * width; // start looking 10 pixels down from current position
      while (i < width * height && i < init + width * 50) {
        if (pixelData[i] == 255) {
          difference[tid] = (i - init) / width;
          count[tid] = 1;
          break;
        } else {
          difference[tid] = 0;
          count[tid] = 0;
        }
        i = i + width;
      }
    } else {
      difference[tid] = 0;
      count[tid] = 0;
    }
    /**
    __syncThreads();
    //reduction for sum
    for (unsigned int s = 1; s < blockDim.x; s*=2){
      int index = 2*s*tid;
      if (index < blockDim.x){
        difference[index] += difference[index+s];
      }
      __syncTreads();
    }
    if (tid ==0)
    printf("%d__%d  ",
    }
    **/
  }
}

__global__ void spacing(const unsigned char *pixelData, int width, int height) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  if (tid < width * height) {
    if (pixelData[tid] == 255) {
      int init = tid;
      tid = init + 10 * width;
      // looking below by up to 20 pixels
      while (tid < width * height && tid < init + width * 50) {
        if (255 - pixelData[tid] == 0) {
          int difference = (tid - init) / width;
          printf("%d  ", difference);
          break;
        }
        tid = tid + width;
        difference = 0;
      }
    } else {
      difference = 0;
    }
  }
}