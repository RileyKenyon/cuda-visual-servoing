#include "imgProc.hpp"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>

static constexpr unsigned int Frames = 120;
static constexpr unsigned int numThreads = 1024; // good number for multiple of 32

int main(int argc, char const *argv[]) {
  // Initialize timer settings
  float calcTimer = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Initialize capture source
  cv::Mat img;
  auto cap = (argc == 2) ? cv::VideoCapture(argv[1]) : cv::VideoCapture(1);
  if (!cap.isOpened()) {
    std::runtime_error("Error getting Stream");
  }
  cap >> img;

  cv::imshow("original", img);
  cv::waitKey(0);

  const int imageWidth = img.cols;
  const int imageHeight = img.rows;

  // Do some ROI and calibration to select screen size
  cv::Rect2d r = cv::selectROI(img);
  const int width = r.width;
  const int height = r.height;

  unsigned int numBlocksImage = (imageWidth * imageHeight + numThreads - 1) / numThreads;
  unsigned int numBlocksScreen = (width * height + numThreads - 1) / numThreads;

  // SETUP SETTINGS
  //-------------------------------------------------------------------
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
  std::cout << "Size of image: " << imageInfo[0] << ", " << imageInfo[1] << std::endl;
  std::cout << "Size of ROI: " << screenInfo[0] << ", " << screenInfo[1] << ", " << screenInfo[2] << ", "
            << screenInfo[3] << std::endl;
  cudaMemcpy(matA,
             img.data,
             imageWidth * imageHeight * 3 * sizeof(unsigned char),
             cudaMemcpyHostToDevice); // FOR COPYING ARRAY
  vservo::gpu_grayscale<<<numBlocksImage, numThreads>>>(matA, grayData, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  vservo::screenAllocate<<<numBlocksScreen, numThreads>>>(grayData, screenData, imageInfo, screenInfo);
  cudaDeviceSynchronize();
  vservo::edgeFind<<<numBlocksScreen, numThreads>>>(screenData, prevArr, width, height);
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
      vservo::gpu_grayscale<<<numBlocksImage, numThreads>>>(matA, grayData, imageWidth, imageHeight);
      cudaDeviceSynchronize(); // sync threads and cpy mem
      vservo::screenAllocate<<<numBlocksScreen, numThreads>>>(grayData, screenData, imageInfo, screenInfo);
      cudaDeviceSynchronize();
      vservo::edgeFind<<<numBlocksScreen, numThreads>>>(screenData, edge, width, height);
      cudaDeviceSynchronize();
      vservo::addArr<<<numBlocksScreen, numThreads>>>(edge, prevArr, output, width, height);
      cudaDeviceSynchronize();
      vservo::spacing<<<numBlocksScreen, numThreads>>>(output, width, height);
      cudaDeviceSynchronize();
      // std::cout << screenData;
      cv::Mat build(cv::Size(width, height), CV_8UC1, output);
      // cv::Mat build(cv::Size(imageWidth,imageHeight),CV_8UC1,grayData);
      memcpy(prevArr, edge, width * height * sizeof(unsigned char));

      // saving
      // cv::Mat videoFrameGray(cv::Size(width,height),CV_8UC3);
      cv::Mat videoFrameGray;
      cv::cvtColor(build, videoFrameGray, cv::COLOR_GRAY2BGR); // 3 array of grayscale for saving to file
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
