#include "camera.hpp"

namespace camera
{
  GenericCamera::GenericCamera()
  : cv::VideoCapture(0)
  {
  }
  
  GenericCamera::~GenericCamera() = default;

  std::string GenericCamera::get_tegra_gst_pipeline_str()
  {
      return ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(mu32_captureWidth) + \
        ", height=(int)" + std::to_string(mu32_captureHeight) + ",format=(string)NV12, \
        framerate=(fraction)" + std::to_string(mu8_captureFPS) + "/1 ! nvvidconv ! video/x-raw, format=(string)BGRx \
        ! videoconvert !  appsink");
  }
}