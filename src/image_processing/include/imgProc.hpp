///-----------------------------------------------------------------------------
/// @file imgProc.hpp
///
/// @author Riley Kenyon (rike2277@colorado.edu)
/// @brief
///
/// @date 2024-01-27
///-----------------------------------------------------------------------------
#ifndef IMGPROC_HPP
#define IMGPROC_HPP
#include <opencv2/highgui/highgui.hpp>

/// @brief Convert image to grayscale using the CPU
/// @param[in] img RGB image
/// @param[out] grayscale Grayscale image
void convert_grayscale_cpu(const cv::Mat *img, cv::Mat *grayscale);

/// @brief Convert image to grayscale using the GPU
/// @param[in] matA RBG image
/// @param[out] matG Grayscale image
/// @param width Width of the image
/// @param height Height of the image
__global__ void gpu_grayscale(const unsigned char *matA, unsigned char *matG, int width, int height);

/// @brief Allocate the screen
/// @param[in] originalImage The original image
/// @param[out] screenImage The output screen
/// @param[in] imageInfo Information regarding the image
/// @param[in] screenInfo Information regarding the screen
__global__ void screenAllocate(const unsigned char *originalImage,
                               unsigned char *screenImage,
                               const int *imageInfo,
                               const int *screenInfo);

#endif // IMGPROC_HPP
