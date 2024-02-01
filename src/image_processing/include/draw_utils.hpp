///-----------------------------------------------------------------------------
/// @file draw_utils.hpp
///
/// @author Riley Kenyon (rike2277@colorado.edu)
/// @brief Additional utilities to assist in the drwawing process
///
/// @date 2024-01-31
///-----------------------------------------------------------------------------
#ifndef DRAW_UTILS_HPP
#define DRAW_UTILS_HPP

#include <opencv2/opencv.hpp>

namespace vservo {

void draw_circle(cv::Mat img, cv::Point center) {
  cv::circle(img, center, 10, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_8);
};

double annotate_with_circle(unsigned char *edge, double velocity, cv::Mat img) {
  auto trigger = get_channel_middle_px(width, height);
  for (int i = 0; i < trigger.size(); i++) {
    int pixelCount = 0;
    int index = 0;
    auto width = img.cols;
    auto height = img.rows;

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

std::arrary<int, 4> get_channel_middle_px(int width, int height) {
  const auto offset = width / 8;
  return {offset, offset + (width / 4), offset + (width / 2), offset + (width * (3 / 4))};
};

} // namespace vservo

#endif // DRAW_UTILS_HPP
