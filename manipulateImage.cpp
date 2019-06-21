#include <opencv/cv.hpp>
#include <iostream>

int main()
{
  //cv::Mat img();
  cv::Mat img;
  img = cv::imread("example.jpeg");
  //img = img(Range::all(),Range::all(),0);
  cv::imshow("example.jpeg",img);
  cv::waitKey();
}
