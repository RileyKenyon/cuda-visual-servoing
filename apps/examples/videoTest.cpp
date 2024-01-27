#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;

int main() {
  cv::Mat img; // (640,480,CV_8UC3);
  cv::Mat gray;
  const char *gst;
  gst =
      "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)30/1 ! 		  nvvidconv flip-method=6 ! video/x-raw, format=(string)I420 ! \
	  videoconvert ! video/x-raw, format=(string)BGR ! \
	  appsink";
  // cv::VideoCapture cap(0); // USB Camera
  cv::VideoCapture cap(gst);
  if (!cap.isOpened()) {
    cout << "Error getting stream" << endl;
    return -1;
  }

  cap >> img;
  // cout << (img.rows);
  cout << img.data;
  // cout << (img.cols);
  for (;;) {
    cap >> img; // newframe from camera

    cvtColor(img, gray, CV_RGB2GRAY); // perform with GPU shifts from RGB 2 Gray
    cv::imshow("grayscale", img);
    char c = cv::waitKey(30);
    if (c == ' ')
      break;
    // std::cout<<img.channels();
  }
  cap.release();
  // video.release();
  // destroyAllWindows();
  return 0;
}
