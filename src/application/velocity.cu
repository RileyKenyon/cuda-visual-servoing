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
__global__ void addArr(unsigned char *arrA, unsigned char *arrB, unsigned char *output, int width, int height) {
  int tid, stride;
  tid = blockIdx.x * blockDim.x + threadIdx.x;
  stride = blockDim.x * gridDim.x;
  while (tid < width * height) {
    output[tid] = arrA[tid] + arrB[tid];
    tid = tid + stride;
  }
}

__global__ void spacing(unsigned char *pixelData, int width, int height) {
  int tid, stride, difference;
  tid = blockIdx.x * blockDim.x + threadIdx.x;
  stride = blockDim.x * gridDim.x;
  if (tid < width * height) {
    if (pixelData[tid] == 255) {
      int init = tid;
      tid = init + 10 * width;
      // looking below by up to 20 pixels
      while (tid < width * height && tid < init + width * 50) {
        if (255 - pixelData[tid] == 0) {
          difference = (tid - init) / width;
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
  cv::VideoCapture cap("color.avi");
  // cv::VideoCapture cap(1); // webcam
  if (!cap.isOpened()) {
    printf("Error getting Stream \n");
  }
  cap >> img;
  // cv::imshow("original",img);
  // cv::waitKey(0);
  int imageWidth = img.cols;
  int imageHeight = img.rows;
  printf("Resolution: %d x %d \n", imageWidth, imageHeight);
  // Do some ROI and calibration to select screen size
  cv::Rect2d r = cv::selectROI(img);
  int width = r.width;
  int height = r.height;
  // unsigned char *screenData = (unsigned char *) malloc(width*height*sizeof(unsigned char));

  unsigned int numThreads, numBlocksImage, numBlocksScreen;
  numThreads = 1024; // good number for multiple of 32
  numBlocksImage = (imageWidth * imageHeight + numThreads - 1) / numThreads;
  numBlocksScreen = (width * height + numThreads - 1) / numThreads;
  /** Write to file
    std::ofstream dataFile;
    dataFile.open ("output.txt");
    for (int k = 0; k<3; k++){
      for (int j = 100*(k+1)*width; j<((100*(k+1))+1)*width; j++){
         int output = lineData[j] - '0';
         dataFile << output;
         if (j != ((100*(k+1))+1)*width-1){ //last element in row
           dataFile << ",";
         }
         lineData[j] = 0;
      }
    dataFile << "\n";
    }
    dataFile.close();
    **/
  // SETUP SETTINGS
  //-------------------------------------------------------------------
  // Configure blocks and threads for GPU
  /**
  unsigned int numThreads, numBlocks;
  numThreads = 1024; // good number for multiple of 32
  numBlocks = (width*height + numThreads - 1)/numThreads;
  **/
  // Allocate device and host
  unsigned char *matA, *screenData, *grayData, *edge, *prevArr, *output;
  int *imageInfo, *screenInfo;
  cudaMallocManaged(&matA, sizeof(unsigned char) * imageWidth * imageHeight * 3);
  cudaMallocManaged(&grayData, sizeof(unsigned char) * imageWidth * imageHeight);
  cudaMallocManaged(&screenData, sizeof(unsigned char) * width * height);
  cudaMallocManaged(&edge, sizeof(unsigned char) * width * height);
  cudaMallocManaged(&prevArr, sizeof(unsigned char) * width * height);
  cudaMallocManaged(&output, sizeof(unsigned char) * width * height);
  cudaMallocManaged(&imageInfo, sizeof(int) * 2);
  cudaMallocManaged(&screenInfo, sizeof(int) * 4);

  // GPU CALCULATION
  //-----------------------------------------------------------------
  // Initial assignment to previous arr
  int imageInfoHost[2] = {imageWidth, imageHeight};
  int screenInfoHost[4] = {r.x, r.y, r.width, r.height};
  cudaMemcpy(imageInfo, imageInfoHost, 2 * sizeof(int), cudaMemcpyHostToDevice);   // FOR COPYING ARRAY
  cudaMemcpy(screenInfo, screenInfoHost, 4 * sizeof(int), cudaMemcpyHostToDevice); // FOR COPYING ARRAY
  printf("Size of image: %d, %d \n", imageInfo[0], imageInfo[1]);
  printf("Size of ROI: %d,%d,%d,%d \n", screenInfo[0], screenInfo[1], screenInfo[2], screenInfo[3]);
  cudaMemcpy(matA,
             img.data,
             imageWidth * imageHeight * 3 * sizeof(unsigned char),
             cudaMemcpyHostToDevice); // FOR COPYING ARRAY
  gpu_grayscale<<<numBlocksImage, numThreads>>>(matA, grayData, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  screenAllocate<<<numBlocksScreen, numThreads>>>(grayData, screenData, imageInfo, screenInfo);
  cudaDeviceSynchronize();
  edgeFind<<<numBlocksScreen, numThreads>>>(screenData, prevArr, width, height);
  cudaDeviceSynchronize();
  char c; // for waitkey
  cv::Mat test(cv::Size(width, height), CV_8UC1, screenData);
  cv::imshow("GPU", test);
  c = cv::waitKey(0);

  // Video Writer
  cv::VideoWriter writer;
  int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
  double fps = 30;
  std::string filename = "./converted.avi";
  writer.open(filename, codec, fps, test.size(), 1); // boolean at end is color
  if (!writer.isOpened()) {
    printf("Unable to Open Video\n");
    return -1;
  }

  // Converter Loop
  for (;;) {
    cudaEventRecord(start);
    for (int j = 0; j < Frames; j++) {
      // capture and calculations
      cap >> img;
      cudaMemcpy(matA,
                 img.data,
                 imageWidth * imageHeight * 3 * sizeof(unsigned char),
                 cudaMemcpyHostToDevice); // FOR COPYING ARRAY
      gpu_grayscale<<<numBlocksImage, numThreads>>>(matA, grayData, imageWidth, imageHeight);
      cudaDeviceSynchronize(); // sync threads and cpy mem
      screenAllocate<<<numBlocksScreen, numThreads>>>(grayData, screenData, imageInfo, screenInfo);
      cudaDeviceSynchronize();
      edgeFind<<<numBlocksScreen, numThreads>>>(screenData, edge, width, height);
      cudaDeviceSynchronize();
      addArr<<<numBlocksScreen, numThreads>>>(edge, prevArr, output, width, height);
      cudaDeviceSynchronize();
      spacing<<<numBlocksScreen, numThreads>>>(output, width, height);
      cudaDeviceSynchronize();
      // std::cout << screenData;
      cv::Mat build(cv::Size(width, height), CV_8UC1, output);
      // cv::Mat build(cv::Size(imageWidth,imageHeight),CV_8UC1,grayData);
      memcpy(prevArr, edge, width * height * sizeof(unsigned char));

      // saving
      // cv::Mat videoFrameGray(cv::Size(width,height),CV_8UC3);
      cv::Mat videoFrameGray;
      cv::cvtColor(build, videoFrameGray, CV_GRAY2BGR); // 3 array of grayscale for saving to file
      // cv::imshow("savingExample",videoFrameGray);
      // c = cv::waitKey(0);
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
  cudaFree(edge);
  cudaFree(prevArr);
  cudaFree(output);
}
