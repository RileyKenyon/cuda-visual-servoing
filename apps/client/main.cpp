#include "config.h"
#include "errors.h"
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  cv::VideoCapture *cam = nullptr;
  cv::Mat frameIn;

  // Parse input arguments
  if (argc < 2) {
    cam = new cv::VideoCapture(0);
  } else {
    // report out the version
    std::cout << argv[0] << " Version " << VSERVO_VERSION_MAJOR << "." << VSERVO_VERSION_MINOR << std::endl;
    std::cout << "Usage: " << argv[0] << " number" << std::endl;
    return vservo::ARG_ERR;
  }

  // Video Capture
  if (cam != nullptr) {
    while (cam->isOpened()) {
      // error handling
      if (!cam->read(frameIn)) {
        std::cout << "Capture read error" << std::endl;
        break;
      }

      // show the image
      else {
        cv::imshow("MyCameraPreview", frameIn);
        if (cv::waitKey(1) >= 0) {
          printf("Exiting.");
          break;
        }
      }
    }
    cam->release();
    delete cam;
  }

  return vservo::NO_ERR;
}