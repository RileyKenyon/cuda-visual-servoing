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

static constexpr unsigned int numThreads = 1024; // good number for multiple of 32

int main(int argc, char const *argv[]) {
  // Initialize timer settings
  float calcTimer = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

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
  std::cout << "Resolution: " << imageWidth << "x" << imageHeight << std::endl;

  // Do some ROI and calibration to select screen size
  cv::Rect2d r = cv::selectROI(img);
  int width = r.width;
  int height = r.height;
  // unsigned char *screenData = (unsigned char *) malloc(width*height*sizeof(unsigned char));

  unsigned int numBlocksImage = (imageWidth * imageHeight + numThreads - 1) / numThreads;
  unsigned int numBlocksScreen = (width * height + numThreads - 1) / numThreads;
  std::cout << "Number of Blocks Cropped: " << numBlocksScreen << std::endl;

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
  // TEST FOR vservo::spacing
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
      vservo::gpu_grayscale<<<numBlocksImage, numThreads>>>(matA, grayData, imageWidth, imageHeight);
      cudaDeviceSynchronize(); // sync threads and cpy mem
      vservo::screenAllocate<<<numBlocksScreen, numThreads>>>(grayData, screenData, imageInfo, screenInfo);
      cudaDeviceSynchronize();
      vservo::edgeFind<<<numBlocksScreen, numThreads>>>(screenData, edge, width, height);
      cudaDeviceSynchronize();
      vservo::addArr<<<numBlocksScreen, numThreads>>>(edge, prevArr, output, width, height);
      cudaDeviceSynchronize();
      vservo::spacing<<<numBlocksScreen, numThreads>>>(output, difference, count, width, height);
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
