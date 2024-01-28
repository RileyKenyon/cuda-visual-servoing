#include "imgProc.hpp"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>

#define NUM 10000
#define Frames 120
#define LENGTH(x) (sizeof(x) / sizeof((x)[0]))
// GPU KERNELS
//----------------------------------------------------------------
__global__ void spacing(unsigned char *pixelData,
                        unsigned int *difference,
                        unsigned int *count,
                        unsigned int width,
                        int height) {
  // extern__shared__int difference[];
  // extern__shared__int count;
  int tid, i, stride;
  tid = blockIdx.x * blockDim.x + threadIdx.x;
  stride = blockDim.x * gridDim.x;
  if (tid < width * height) {
    if (pixelData[tid] == 255) {
      int init = tid;
      i = init + 10 * width; // start looking 10 pixels down from current position
      while (i < width * height && i < init + width * 50) {
        if (255 - pixelData[i] == 0) {
          difference[tid] = (i - init) / width;
          count[tid] = 1;
          // printf("%d  ",difference[tid]);
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
  int imageWidth = img.cols;
  int imageHeight = img.rows;
  // cv::imshow("original",img);
  // cv::waitKey(0);
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
  printf("Number of Blocks Cropped: %d	", numBlocksScreen);
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
  unsigned int *difference, *count;
  cudaMallocManaged(&matA, sizeof(unsigned char) * imageWidth * imageHeight * 3);
  cudaMallocManaged(&grayData, sizeof(unsigned char) * imageWidth * imageHeight);
  cudaMallocManaged(&screenData, sizeof(unsigned char) * width * height);
  cudaMallocManaged(&edge, sizeof(unsigned char) * width * height);
  cudaMallocManaged(&prevArr, sizeof(unsigned char) * width * height);
  cudaMallocManaged(&output, sizeof(unsigned char) * width * height);
  cudaMallocManaged(&imageInfo, sizeof(int) * 2);
  cudaMallocManaged(&screenInfo, sizeof(int) * 4);
  cudaMallocManaged(&difference, sizeof(unsigned int) * width * height);
  cudaMallocManaged(&count, sizeof(unsigned int) * width * height);

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
  // TEST FOR SPACING
  for (int j = 0; j < width * height; j++) {
    screenData[j] = 255;
    if (j < 5 * width && j > 4 * width) {
      screenData[j] = 0;
    }
    if (j < 25 * width && j > 24 * width) {
      screenData[j] = 0;
    }
  }
  cv::imshow("GPU", test);
  c = cv::waitKey(0);
  cv::imwrite("20px_Distance.png", test);

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
      spacing<<<numBlocksScreen, numThreads>>>(output, difference, count, width, height);
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
      for (int i = 0; i < width * height; i++) {
        difference[0] += difference[i];
        count[0] += count[i];
      }
      printf("Velocity:%d \n", difference[0] / count[0]);
      c = cv::waitKey(1);
      writer.write(videoFrameGray); // write to video file

      cv::imshow("GPU", build);
      c = cv::waitKey(1);
      if (c == 'p')
        cv::imwrite("differenceProgress.png", build);
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
