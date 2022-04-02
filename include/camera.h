#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#ifndef TEGRA_CAMERA_H_
#define TEGRA_CAMERA_H_

#define CAMERA_WIDTH    (960U)
#define CAMERA_HEIGHT   (616U)
#define CAMERA_FPS      (120U)

std::string get_tegra_gst_pipeline(const uint32_t width, const uint32_t height, const uint8_t fps);

#endif  // TEGRA_CAMERA_H_