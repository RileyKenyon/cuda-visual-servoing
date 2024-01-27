#include <iostream>
#include <opencv2/opencv.hpp>
//#include </usr/include/gstreamer-1.0/gst/gst.h>
int main() {
  const char *gst;
  // gst = "nvcamerasrc ! 'video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)I420,
  // framerate=(fraction)60/1' ! nvvidconv ! 'video/x-raw(memory:NVMM), format=(string)I420' ! nvoverlaysink -e"; gst =
  // "nvcamerasrc ! 'video/x-raw, format=(string)BGR, framerate=(fraction)[ 0/1, 2147483647/1 ], width=(int)2592,
  // height=(int)1944, interlace-mode=(string)progressive, pixel-aspect-ratio=(fraction)1/1, colorimetry=(string)sRGB' !
  // nvoverlaysink -e"; gst = "imxv4l2videosrc device = \"/dev/video0\" ! videoconvert ! appsink";
  //  open the camera with the gst string
  cv::VideoCapture cap(1); // + cv::CAP_GSTREAMER
  // cap.open(0); //  + cv::CAP_GSTREAMER
  //  error handling
  if (!cap.isOpened()) {
    std::cout << "Failed to open camera." << std::endl;
    return -1;
  }

  unsigned int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
  unsigned int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
  unsigned int pixels = width * height;
  std::cout << "Frame size : " << width << " x " << height << ", " << pixels << " Pixels " << std::endl;

  cv::namedWindow("MyCameraPreview", CV_WINDOW_AUTOSIZE);

  // create Mat item with width, height from camera and make it color
  cv::Mat frame_in(width, height, CV_8UC3);

  while (1) {
    // error handling
    if (!cap.read(frame_in)) {
      std::cout << "Capture read error" << std::endl;
      break;
    }

    // show the image
    else {
      cv::imshow("MyCameraPreview", frame_in);
      if (cv::waitKey(1) >= 0) {
        printf("Exiting.");
        break;
      }
    }
  }

  cap.release();

  return 0;
}
