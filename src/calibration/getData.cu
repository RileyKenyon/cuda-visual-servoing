#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>

#define NUM 10000
#define Frames 120
// MAIN FUNCTION
//-----------------------------------------------------------------
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
  img = cv::imread("Screen.png");
  // cv::imshow("original",img);
  unsigned char *cpuData = (unsigned char *)malloc(img.cols * img.rows * sizeof(unsigned char));
  for (int i = 0; i < img.rows * img.cols; i++) {
    cpuData[i] = img.data[3 * i] * 0.07 + img.data[3 * i + 1] * 0.72 + img.data[3 * i + 2] * 0.21;
  }
  cv::Mat reconstruction(cv::Size(img.cols, img.rows), CV_8UC1, cpuData);
  cv::imshow("cpu_gray", reconstruction);
  cv::waitKey(0);
  // select ROI
  cv::Rect2d r = cv::selectROI(reconstruction);
  // cv::Mat cropped = reconstruction(r);
  // cv::imshow("cropped",cropped);
  // cv::waitKey();
  // CALCULATION SETTINGS
  //-------------------------------------------------------------------
  // Size of windows
  int width = r.width;
  int height = r.height;
  // printf("Initial value %u \n", cropped.data[1]);
  printf("Resolution: %d x %d \n", width, height);
  // Configure blocks and threads for GPU
  unsigned int numThreads, numBlocks;
  numThreads = 1024;                                          // good number for multiple of 32
  numBlocks = (width * height + numThreads - 1) / numThreads; // make so at maximum only one additional block

  // Allocate device and host
  // unsigned char *matA, *grayData;
  // cudaMallocManaged(&matA,sizeof(unsigned char)*width*height*3);
  // cudaMallocManaged(&grayData,sizeof(unsigned char)*width*height);
  unsigned char *lineData = (unsigned char *)malloc(width * height * sizeof(unsigned char));
  // memcpy(lineData,cropped.data,width*height*sizeof(unsigned char));

  // Edge detection
  cv::Mat kernel = (cv::Mat_<int>(3, 2) << 1, 1, 0, 0, 0, 0);
  cv::Mat outputGPU;

  /**
  // GPU CALCULATION
  //-----------------------------------------------------------------
    //cudaEventRecord(start);
    for (;;){
      cudaEventRecord(start);
      char c;
      for (int j = 0; j< Frames; j++){
        cap >> img;
        cudaMemcpy(matA, img.data, width*height*3* sizeof(unsigned char), cudaMemcpyHostToDevice); // NEED THIS LINE FOR
  COPYING ARRAY gpu_grayscale<<<numBlocks,numThreads>>>(matA,grayData,width,height); cudaDeviceSynchronize(); // sync
  threads and transfer memory
      // Show grayscale image
        //cv::Mat outputGPU(cv::Size(width,height),CV_8UC1,grayData);
        cv::Mat build(cv::Size(width,height),CV_8UC1,grayData); // NEED TO CHANGE TO GPU DATA IF USING CUDAMALLOC AND
  NOT CUDAMALLOCMANAGED
        // implement edge detect
        cv::morphologyEx(build,outputGPU,cv::MORPH_HITMISS,kernel);
        //cv::rotate(outputGPU,outputGPU,cv::ROTATE_90_COUNTERCLOCKWISE); // for rotating recorded video
        cv::imshow("GPU",outputGPU);
        c = cv::waitKey(1);
        if (c ==' ')
          break;
      }
      if (c ==' ')
        break;
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&calcTimer,start, stop);
      printf("FPS GPU: %f \n",Frames/calcTimer*1000);
      calcTimer = 0;
    }
  **/
  // CPU CALCULATION
  //--------------------------------------------------------------------
  // for (;;){
  cudaEventRecord(start);
  char c;
  // for (int j = 0; j<Frames; j++){
  for (int y = r.y; y < (r.y + r.height); y++) {
    // printf("-----------------------------------------------------\n");
    for (int x = r.x; x < (r.x + r.width); x++) {
      int index = (y * width + x) - (r.y * width + r.x);
      int roi = y * img.cols + x;
      // printf("new Mat: %d Old Mat: %d\n",index,roi);
      // printf("width: %g height: %g\n",r.width,r.height);
      // printf("y_initial: %g x_initial: %g\n",r.y,r.x);
      // cv::waitKey(0);
      lineData[index] = reconstruction.data[roi];
      // reconstruction.data[roi] = 0;
      // imshow("update",reconstruction);
      // cv::waitKey(1);
      /**
       if (i < width*height){
         lineData[i] = 0;
       }
      **/
    }
  }
  // Write to file
  std::ofstream dataFile;
  dataFile.open("output.txt");
  // unsigned char *outputData = (unsigned char *) malloc(r.width*r.height*sizeof(unsigned char));
  // dataFile << "[";
  for (int k = 0; k < 3; k++) {
    for (int j = 100 * (k + 1) * width; j < ((100 * (k + 1)) + 1) * width; j++) {
      int output = lineData[j] - '0';
      dataFile << output;
      if (j != ((100 * (k + 1)) + 1) * width - 1) {
        dataFile << ",";
      }
      lineData[j] = 0;
    }
    dataFile << "\n";
  }
  // dataFile << "]";
  dataFile.close();

  cv::Mat grab(cv::Size(width, height), CV_8UC1, lineData);
  cv::imshow("cpu_gray", grab);
  cv::waitKey(0);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&calcTimer, start, stop);
  printf("FPS CPU: %f\n", Frames / calcTimer * 1000);
  calcTimer = 0;
  //}
  // CLOSEOUT
  //--------------------------------------------------------------------------
  // cudaFree(matA);
  // cudaFree(grayData);
  free(lineData);
  free(cpuData);
  return 0;
}
