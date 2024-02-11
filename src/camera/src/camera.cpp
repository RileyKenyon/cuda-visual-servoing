///-----------------------------------------------------------------------------
/// @file camera.cpp
///
/// @author Riley Kenyon (rike2277@colorado.edu)
/// @brief Implementation of the tegra gstreamer pipeline
///
/// @date 2024-02-11
///-----------------------------------------------------------------------------
#include "camera.hpp"

namespace camera {
GenericCamera::GenericCamera() : cv::VideoCapture(0) {}

GenericCamera::~GenericCamera() = default;

std::string GenericCamera::get_tegra_gst_pipeline_str() {
  return ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(mu32_captureWidth) +
          ", height=(int)" + std::to_string(mu32_captureHeight) + ",format=(string)NV12, \
        framerate=(fraction)" +
          std::to_string(mu8_captureFps) + "/1 ! nvvidconv ! video/x-raw, format=(string)BGRx \
        ! videoconvert !  appsink");
}
} // namespace camera