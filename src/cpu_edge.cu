#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <string>

#define NUM 10000
#define Frames 120
// MAIN FUNCTION
//-----------------------------------------------------------------
//using namespace cv;
int main()
{
  // Initialize timer settings
  float calcTimer = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float CPUtimer;

// ORIGINAL IMAGE
//-------------------------------------------------------------------
  // Get initial image and print
  cv::Mat img;
  //img = cv::imread("pineapple.jpeg");
  //cv::VideoCapture cap("example.mp4"); // replace with 1 if using webcam
  cv::VideoCapture cap(1);
  if (!cap.isOpened()){
    printf("Error getting Stream \n");
  }
  cap >> img;

  // Size of windows  
  int width = img.cols;
  int height = img.rows;
  printf("Resolution: %d x %d \n",width, height);

  unsigned char *cpuData = (unsigned char *) malloc(img.cols*img.rows*sizeof(unsigned char));
  unsigned char *edge = (unsigned char *) malloc(img.cols*img.rows*sizeof(unsigned char));

  for (int i = 0; i < img.rows*img.cols; i++){
    cpuData[i] = img.data[3*i]*0.07 + img.data[3*i+1]*0.72 + img.data[3*i+2]*0.21;
  }
  cv::Mat reconstruction(cv::Size(img.cols,img.rows),CV_8UC1,cpuData);
  // select ROI
  //cv::rotate(reconstruction,reconstruction,cv::ROTATE_90_COUNTERCLOCKWISE); //for rotating video
  cv::Rect2d r = cv::selectROI(reconstruction);
  // Create calibration lines
    for( int k = 0; k<3; k++){
      for(int j = 0; j < (r.height); j++){
        int number = r.width*j + 100*(k+1); // offset by 100  
	cpuData[number] = 0;        
      }    
    }
  cv::Mat calibration(cv::Size(img.cols,img.rows),CV_8UC1,cpuData);
  cv::imshow("Calibration Lines",calibration);
  cv::waitKey(0);

// CPU CALCULATION
//--------------------------------------------------------------------
for (;;){
    cudaEventRecord(start);
    char c;
    for (int j = 0; j<Frames; j++){
      cap >> img;
      // cpu grayscale
      for (int i = 0; i < width*height; i++){
        cpuData[i] = img.data[3*i]*0.07 + img.data[3*i+1]*0.72 + img.data[3*i+2]*0.21;
	if (i > width){
          if (cpuData[i] > 150 && cpuData[i-width] < 150){ //saturate to either 255 or 0 - for pixel testing
            edge[i] = 255;
          } else {
            edge[i] = 0;
          }
	}
      }
      cv::Mat reconstruction(cv::Size(img.cols,img.rows),CV_8UC1,edge);
      //cv::Mat outputCPU;
      //cv::morphologyEx(reconstruction,outputCPU,cv::MORPH_HITMISS,kernel); 
      //cv::rotate(reconstruction,reconstruction,cv::ROTATE_90_COUNTERCLOCKWISE); // for rotating recorded video
      cv::imshow("cpu_gray",reconstruction);
      c = cv::waitKey(1);
      if (c ==' ')
        break;
    }
    if (c ==' ')
      break;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&calcTimer,start, stop);
    printf("FPS CPU: %f\n",Frames/calcTimer*1000);
    calcTimer = 0;
  }
// CLOSEOUT
//--------------------------------------------------------------------------
  free(cpuData);
}
