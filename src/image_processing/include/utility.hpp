///-----------------------------------------------------------------------------
/// @file utility.hpp
///
/// @author Riley Kenyon (rike2277@colorado.edu)
/// @brief Additional utilities to assist in the drawing and logging
///
/// @date 2024-01-31
///-----------------------------------------------------------------------------
#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <fstream>
#include <opencv2/opencv.hpp>

namespace vservo {

/// @brief Annotate an image with a circle
/// @param[in,out] img image to annotate
/// @param[in] center center of the circle
void draw_circle(cv::Mat img, const cv::Point center) {
  cv::circle(img, center, 10, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_8);
};

/// @brief Get the channel midde pixels
/// @param width width of the image
/// @param height height of the image
/// @return an array of 4 with the column index of the column middle
std::array<int, 4> get_channel_middle_px(int width, int height) {
  const auto offset = width / 8;
  return {offset, offset + (width / 4), offset + (width / 2), offset + (width * (3 / 4))};
};

/// @brief Annotate an image with a circle
/// @param[in] edge edge data
/// @param velocity estimated velocity of the image if moving
/// @param[in,out] img The image to annotate
void annotate_with_circle(unsigned char *edge, double velocity, cv::Mat img);

/// @brief Log the input data to a file with a given offset
/// @param fname the log file
/// @param[in] data pointer to the data
/// @param width width of the data
/// @param height height of the data
/// @param offset offset of the data
void log_to_file(const std::string &fname, const unsigned int *data, int width, int height, int offset = 100);

/// @brief Create calibration lines on the passed in data
/// @param[in, out] data The data to draw calibration lines
/// @param width width of the input data
/// @param height height of the input data
/// @param offset offset of the input data
void create_calibration_lines(unsigned int *data, int width, int height, int offset = 100);

} // namespace vservo

#endif // UTILITY_HPP
