#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <string>

// MAIN FUNCTION
//-----------------------------------------------------------------
//using namespace cv;
int main()
{
// ORIGINAL IMAGE
//-------------------------------------------------------------------
  // Get initial image and print
  cv::Mat img;
  cv::VideoCapture cap(1);
  if (!cap.isOpened()){
    printf("Error getting Stream \n");
  }
  cap >> img;
  cv::imshow("original",img);
  char c = cv::waitKey(0);
  if (c=='q')
    return -1;
  int width = img.cols;
  int height = img.rows;
  printf("Resolution: %d x %d \n",width, height);

  // Video Writer
  cv::VideoWriter writer;
  int codec = cv::VideoWriter::fourcc('M','J','P','G');
  double fps = 30;
  std::string filename = "./Test1.avi";
  writer.open(filename,codec,fps,img.size(),1);
  if (!writer.isOpened()){
    printf("Unable to Open Video\n");
    return -1;
  }
  for (;;){
      cap >> img;
      writer.write(img);  // write to video file
      cv::imshow("GPU",img);
      c = cv::waitKey(1);
      if (c ==' ')
        break; 
  }
  return 0;
}
