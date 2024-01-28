#include "imgProc.hpp"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>

#define NUM 10000
#define Frames 120

// GPU KERNELS
//----------------------------------------------------------------
__global__ void edgeFind(unsigned char *grayData, unsigned char *edge, int width, int height) {
  int tid, stride;
  tid = blockIdx.x * blockDim.x + threadIdx.x;
  stride = blockDim.x * gridDim.x;

  while (tid < width * height) {
    if (tid > 2 * width) {
      if (grayData[tid] > 150 && grayData[tid - width] < 150 && grayData[tid - 2 * width] < 150 &&
          grayData[tid - 2] > 150 && grayData[tid - 1] > 150) {
        edge[tid] = 255; // set to white
      } else {
        edge[tid] = 0;
      }
    }
    tid = tid + stride;
  }
}
// MAIN FUNCTION
//-----------------------------------------------------------------
// using namespace cv;
int main() {
  // Initialize timer settings
  float calcTimer = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // float GPUtimer, CPUtimer;

  // ORIGINAL IMAGE
  //-------------------------------------------------------------------
  // Get initial image and print
  cv::Mat img;
  // img = cv::imread("pineapple.jpeg");
  // cv::VideoCapture cap("example.mp4"); // replace with 1 if using webcam
  //  cv::VideoCapture cap(1);
  const int WIDTH = 960;
  const int HEIGHT = 616;
  std::string pipeline = get_tegra_pipeline() cv::VideoCapture
      cap("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)960, height=(int)616,format=(string)NV12, "
          "framerate=(fraction)120/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert !  appsink");

  if (!cap.isOpened()) {
    printf("Error getting Stream \n");
  }
  cap >> img;
  cv::imshow("original", img);
  cv::waitKey();
  int width = img.cols;
  int height = img.rows;
  printf("Resolution: %d x %d \n", width, height);

  // SETUP SETTINGS
  //-------------------------------------------------------------------
  // Configure blocks and threads for GPU
  unsigned int numThreads, numBlocks;
  numThreads = 1024; // good number for multiple of 32
  numBlocks = (width * height + numThreads - 1) / numThreads;

  // Allocate device and host
  unsigned char *matA, *grayData, *edge;
  cudaMallocManaged(&matA, sizeof(unsigned char) * width * height * 3);
  cudaMallocManaged(&grayData, sizeof(unsigned char) * width * height);
  cudaMallocManaged(&edge, sizeof(unsigned char) * width * height);

  // Video Writer
  cv::VideoWriter writer;
  int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
  double fps = 30;
  std::string filename = "./example.avi";
  writer.open(filename, codec, fps, img.size(), 1);
  if (!writer.isOpened()) {
    printf("Unable to Open Video\n");
    return -1;
  }
  // GPU CALCULATION
  //-----------------------------------------------------------------
  // cudaEventRecord(start);
  for (;;) {
    cudaEventRecord(start);
    char c;
    for (int j = 0; j < Frames; j++) {
      cap >> img;
      cudaMemcpy(matA,
                 img.data,
                 width * height * 3 * sizeof(unsigned char),
                 cudaMemcpyHostToDevice); // NEED THIS LINE FOR COPYING ARRAY
      gpu_grayscale<<<numBlocks, numThreads>>>(matA, grayData, width, height);
      cudaDeviceSynchronize(); // sync threads and transfer memory
      // Edge Find
      edgeFind<<<numBlocks, numThreads>>>(grayData, edge, width, height);
      cudaDeviceSynchronize();
      cv::Mat build(cv::Size(width, height), CV_8UC1, edge);
      cv::Mat videoFrameGray; // 3 array of grayscale for saving to file
      cv::cvtColor(build, videoFrameGray, cv::COLOR_GRAY2BGR);
      writer.write(videoFrameGray); // write to video file
      cv::imshow("GPU", build);
      c = cv::waitKey(1);
      if (c == ' ')
        break;
    }
    if (c == ' ')
      break;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&calcTimer, start, stop);
    printf("FPS GPU: %f \n", Frames / calcTimer * 1000);
    calcTimer = 0;
  }

  // CLOSEOUT
  //--------------------------------------------------------------------------
  cudaFree(matA);
  cudaFree(grayData);
}
