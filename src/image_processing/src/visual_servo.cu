///-----------------------------------------------------------------------------
/// @brief Construct a new Visual Servo:: Visual Servo object
///
/// @param width
/// @param height
/// @param layers
///-----------------------------------------------------------------------------
#include "imgProc.hpp"
#include "visual_servo.hpp"
#include <iostream>
#include <stdexcept>
namespace vservo {

VisualServo::VisualServo(unsigned int width, unsigned int height, unsigned char layers)
    : imageWidth(width), imageHeight(height), imageLayers(layers), counter(0), timer(0.0) {
  if (layers > 3) {
    std::runtime_error("An image can contain no more than 3 layers");
  }
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaMallocManaged(&input, sizeof(unsigned char) * width * height * layers);
  cudaMallocManaged(&grayscale, sizeof(unsigned char) * width * height);
  cudaMallocManaged(&output, sizeof(unsigned char) * width * height);
  numBlocks = (width * height + numThreads - 1) / numThreads;
};

VisualServo::~VisualServo() {
  cudaFree(input);
  cudaFree(grayscale);
  cudaFree(output);
}

bool VisualServo::process(unsigned char *data) {
  bool retval = false;
  if (nullptr != data) {
    cudaMemcpy(input, data, imageWidth * imageHeight * imageLayers * sizeof(unsigned char), cudaMemcpyHostToDevice);
    gpu_grayscale<<<numBlocks, numThreads>>>(input, grayscale, imageWidth, imageHeight);
    edgeFind<<<numBlocks, numThreads>>>(grayscale, output, imageWidth, imageHeight);
    get_framerate();
    retval = true;
  }
  return retval;
}

void VisualServo::report_fps(bool shouldReport) { reportFramerate = shouldReport; }
void VisualServo::get_output(unsigned char **data) { *data = output; }

void VisualServo::set_threads(unsigned int threadCount) {
  numThreads = threadCount;
  numBlocks = (imageWidth * imageHeight + numThreads - 1) / numThreads;
}

void VisualServo::get_framerate() {
  if (0 == counter) {
    cudaEventRecord(start);
    counter++;
  }
  if (reportFramerate && (0 == counter % frameWindow)) {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer, start, stop);
    printf("FPS GPU: %f \n", (frameWindow / timer) * 1000.0);
    timer = 0.0f;
    counter = 0;
  } else {
    counter++;
  }
}

} // namespace vservo
