#ifndef TEGRA_CAMERA_H_
#define TEGRA_CAMERA_H_
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <string>

namespace camera
{
  class GenericCamera : public cv::VideoCapture
  {
    public:
      GenericCamera();
      GenericCamera(const uint32_t width, const uint32_t height, const uint8_t FPS);
      ~GenericCamera();

      void open_gst();

    private:
      // Methods
      std::string get_tegra_gst_pipeline();

      // Member Variables
      const uint32_t mu32_captureWidth = 1920;
      const uint32_t mu32_captureHeight = 1080;
      const uint8_t mu8_captureFPS = 30;
  };
}

#endif  // TEGRA_CAMERA_H_