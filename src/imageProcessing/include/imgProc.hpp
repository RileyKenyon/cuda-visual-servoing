#ifndef TEGRA_IMGPROC_H_
#define TEGRA_IMGPROC_H_
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>

/**
 * Grayscale conversion on the CPU and GPU
 **/

void convert_grayscale_cpu(cv::Mat *img, cv::Mat *grayscale);

void convert_grayscale_gpu(cv::Mat *img, cv::Mat *grayscale);

__global__ void gpu_grayscale(unsigned char *matA, unsigned char *matG, int width, int height);