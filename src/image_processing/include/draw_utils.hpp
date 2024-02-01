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
} // namespace vservo

#endif // DRAW_UTILS_HPP
