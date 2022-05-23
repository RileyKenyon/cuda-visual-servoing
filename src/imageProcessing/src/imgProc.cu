#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <string>

void convert_grayscale_cpu(cv::Mat* img, cv::Mat* grayscale)
{
  for (int i = 0; i < img->rows*img->cols; i++){
    grayscale[i] = img->data[3*i]*0.07 + img->data[3*i+1]*0.72 + img->data[3*i+2]*0.21;
  }
}

void convert_grayscale_gpu(cv::Mat* img, cv::Mat* grayscale);


__global__ void gpu_grayscale(  unsigned char *mat, 
                                unsigned char *matG,
                                int width,
                                int height)
{
  //Distance between array elements (i,j)[0] to (i,j)[1] is 1 not width*height
  //thread ID
  int tid;
  tid = blockIdx.x*blockDim.x + threadIdx.x;
  
  //stride lengths
  int stride;
  stride = blockDim.x*gridDim.x;

  // grayscale calculation with strides
  while (tid < width*height){
    matG[tid] = mat[3*tid]*0.07 + mat[3*tid+1]*0.72 + mat[3*tid+2]*0.21;
    if (matG[tid] > 170 ){ //saturate to either 255 or 0 - for pixel testing
      matG[tid] = 255;
    } else {
      matG[tid] = 0;
    }
    tid = tid + stride;
  }
}