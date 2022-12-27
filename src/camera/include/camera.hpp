#ifndef TEGRA_CAMERA_H_
#define TEGRA_CAMERA_H_
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <string>

namespace camera {
class GenericCamera : public cv::VideoCapture {
public:
  GenericCamera();
  ~GenericCamera();

private:
  // Methods
  std::string get_tegra_gst_pipeline_str();

  // Member Variables
  const uint32_t mu32_captureWidth = 1920;
  const uint32_t mu32_captureHeight = 1080;
  const uint8_t mu8_captureFPS = 30;
};
} // namespace camera

#endif // TEGRA_CAMERA_H_