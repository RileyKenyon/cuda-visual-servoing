///-----------------------------------------------------------------------------
/// @file camera.hpp
///
/// @author Riley Kenyon (rike2277@colorado.edu)
/// @brief Video Capture using the tegra gstreamer pipeline
///
/// @date 2024-02-11
///-----------------------------------------------------------------------------
#ifndef CAMERA_HPP
#define CAMERA_HPP
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <string>

namespace camera {
class GenericCamera : public cv::VideoCapture {
public:
  /// @brief Constructor
  GenericCamera();

  /// @brief Destructor
  ~GenericCamera();

private:
  /// @brief Construct a gstreamer pipeline string
  /// @return string of the pipeline
  std::string get_tegra_gst_pipeline_str();
  const uint32_t mu32_captureWidth = 1920;  ///< Default capture width
  const uint32_t mu32_captureHeight = 1080; ///< Default capture height
  const uint8_t mu8_captureFps = 30;        ///< Default capture fps
};
} // namespace camera

#endif // CAMERA_HPP
