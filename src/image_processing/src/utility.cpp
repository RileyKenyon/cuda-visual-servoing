///-----------------------------------------------------------------------------
/// @file utility.cpp
///
/// @author Riley Kenyon (rike2277@colorado.edu)
/// @brief Implementation of utility functions
///
/// @date 2024-02-11
///-----------------------------------------------------------------------------
#include "utility.hpp"

namespace vservo {

void annotate_with_circle(unsigned char *edge, double velocity, cv::Mat img) {
  auto width = img.cols;
  auto height = img.rows;
  auto trigger = get_channel_middle_px(width, height);
  for (int i = 0; i < trigger.size(); i++) {
    int pixelCount = 0;
    int index = 0;

    while (pixelCount < velocity + 2) {
      index = width * height - trigger[i] - (width * pixelCount);
      if (edge[index - 1] == 255 || edge[index] == 255 || edge[index + 1] == 255) {
        vservo::draw_circle(img, cv::Point(trigger[3 - i], height - pixelCount));
        break;
      }
      pixelCount += 1;
    }
  }
};

void log_to_file(const std::string &fname, const unsigned int *data, int width, int height, int offset) {
  std::ofstream dataFile;
  dataFile.open(fname);
  for (int k = 0; k < 3; k++) {
    for (int j = offset * (k + 1) * width; j < ((offset * (k + 1)) + 1) * width; j++) {
      dataFile << (data[j] - '0');
      dataFile << ",";
    }
    dataFile.seekp(dataFile.tellp() - 1L);
    dataFile << "\n";
  }
  dataFile.close();
}

void create_calibration_lines(unsigned int *data, int width, int height, int offset) {
  for (int k = 0; k < 3; k++) {
    for (int j = 0; j < (height); j++) {
      int number = width * j + offset * (k + 1);
      data[number] = 0;
    }
  }
}

std::string get_tegra_gst_pipeline_str(int width, int height, int fps) {
  return ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(width) + ", height=(int)" +
          std::to_string(height) + ",format=(string)NV12, \
        framerate=(fraction)" +
          std::to_string(fps) + "/1 ! nvvidconv ! video/x-raw, format=(string)BGRx \
        ! videoconvert !  appsink");
}

} // namespace vservo