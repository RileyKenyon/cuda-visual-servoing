#include "camera.hpp"

namespace camera
{
  GenericCamera::GenericCamera()
  {
  }

  GenericCamera::GenericCamera(const uint32_t width, const uint32_t height, const uint8_t FPS)
    : mu32_captureWidth(width)
    , mu32_captureHeight(height)
    , mu8_captureFPS(FPS)
  {
  }
  
  GenericCamera::~GenericCamera() = default;

  void GenericCamera::open_gst()
  {
    std::string handle = get_tegra_gst_pipeline();
    this->open(handle);
  }

  std::string GenericCamera::get_tegra_gst_pipeline()
  {
      return ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(mu32_captureWidth) + \
        ", height=(int)" + std::to_string(mu32_captureHeight) + ",format=(string)NV12, \
        framerate=(fraction)" + std::to_string(mu8_captureFPS) + "/1 ! nvvidconv ! video/x-raw, format=(string)BGRx \
        ! videoconvert !  appsink");
  }
}