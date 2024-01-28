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

/// @brief Find the edge of the grayscale image
/// @param[in] grayData Grayscale image
/// @param[out] edge The edge detection
/// @param width Width of the image
/// @param height Height of the image
/// @param threshold The threshold used for edge finding
__global__ void edgeFind(const unsigned char *grayData, unsigned char *edge, int width, int height, int threshold);

/// @brief Add two arrays
/// @param[in] arrA The first image
/// @param[in] arrB The second image
/// @param[out] output The sum of the two images
/// @param width The height of the image
/// @param height The width of the image
__global__ void addArr(const unsigned char *arrA,
                       const unsigned char *arrB,
                       unsigned char *output,
                       int width,
                       int height);

/// @brief Add spacing to the input image
/// @param[in] pixelData input pixel data
/// @param[out] difference difference to the input
/// @param[out] count counter
/// @param width Width of the image
/// @param height Height of the image
__global__ void spacing(const unsigned char *pixelData,
                        unsigned int *difference,
                        unsigned int *count,
                        unsigned int width,
                        int height);

/// @brief Add spacing to the input image
/// @param[in] pixelData input pixel data
/// @param width Width of the image
/// @param height Height of the image
__global__ void spacing(const unsigned char *pixelData, int width, int height);
#endif // IMGPROC_HPP
