#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;

int main() {
  cv::Mat img; // (640,480,CV_8UC3);
  cv::Mat gray;
  cv::vector<Mat> rgb; //(640,480,CV_8UC3);
  // cv::Mat red;
  // cv::Mat green;
  // cv::Mat blue;
  cv::VideoCapture cap(1); // USB Camera

  if (!cap.isOpened()) {
    cout << "Error getting stream" << endl;
    return -1;
  }

  for (;;) {
    cap >> img; // newframe from camera
    cv::split(img, rgb);

    cvtColor(img, gray, CV_RGB2GRAY); // perform with GPU shifts from RGB 2 Gray
    cv::imshow("grayscale", gray);
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
