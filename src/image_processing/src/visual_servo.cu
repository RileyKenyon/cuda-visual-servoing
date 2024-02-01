///-----------------------------------------------------------------------------
/// @brief Construct a new Visual Servo:: Visual Servo object
///
/// @param width
/// @param height
/// @param layers
///-----------------------------------------------------------------------------
#include "imgProc.hpp"
#include "visual_servo.hpp"
#include <fstream>
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

  // Total screen - for some cases, the processing was cropped
  numBlocks = (width * height + kNumThreads - 1) / kNumThreads;
}

VisualServo::~VisualServo() {
  cudaFree(input);
  cudaFree(grayscale);
  cudaFree(output);
}

bool VisualServo::process(unsigned char *data) {
  bool retval = false;
  if (nullptr != data) {
    cudaMemcpy(input, data, imageWidth * imageHeight * imageLayers * sizeof(unsigned char), cudaMemcpyHostToDevice);
    vservo::gpu_grayscale<<<numBlocks, kNumThreads>>>(input, grayscale, imageWidth, imageHeight);
    vservo::edgeFind<<<numBlocks, kNumThreads>>>(grayscale, output, imageWidth, imageHeight);
    get_framerate();
    retval = true;
  }
  return retval;
}

void VisualServo::report_fps(bool shouldReport) { reportFramerate = shouldReport; }
void VisualServo::get_output(unsigned char **data) { *data = output; }

void VisualServo::set_threads(unsigned int threadCount) {
  kNumThreads = threadCount;
  numBlocks = (imageWidth * imageHeight + kNumThreads - 1) / kNumThreads;
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

// void VisualServo::middle(unsigned char **imageIn, unsigned char **imageOut) {
//   cudaMemcpy(imageInfo, imageInfoHost, 2 * sizeof(int), cudaMemcpyHostToDevice);   // FOR COPYING ARRAY
//   cudaMemcpy(screenInfo, screenInfoHost, 4 * sizeof(int), cudaMemcpyHostToDevice); // FOR COPYING ARRAY

//   vservo::gpu_grayscale<<<numBlocksImage, kNumThreads>>>(matA, grayData, imageWidth, imageHeight);
//   vservo::screenAllocate<<<numBlocksScreen, numThreads>>>(grayData, screenData, imageInfo, screenInfo);
//   vservo::edgeFind<<<numBlocksScreen, numThreads>>>(screenData, edge, width, height);
//   vservo::addArr<<<numBlocksScreen, numThreads>>>(edge, prevArr, output, width, height);
//   vservo::spacing<<<numBlocksScreen, numThreads>>>(output, width, height);
// }
// void VisualServo::pixel(unsigned char **imageIn, unsigned char **imageOut) {
//   cv::Mat kernel = (cv::Mat_<int>(3, 2) << 1, 1, 0, 0, 0, 0);
//   vservo::gpu_grayscale<<<numBlocksImage, kNumThreads>>>(matA, grayData, imageWidth, imageHeight);
//   cv::morphologyEx(build, outputGPU, cv::MORPH_HITMISS, kernel);
// }
// void VisualServo::reduction(unsigned char **imageIn, unsigned char **imageOut) {
//   vservo::gpu_grayscale<<<numBlocksImage, kNumThreads>>>(matA, grayData, imageWidth, imageHeight);
//   vservo::screenAllocate<<<numBlocksScreen, kNumThreads>>>(grayData, screenData, imageInfo, screenInfo);
//   vservo::edgeFind<<<numBlocksScreen, kNumThreads>>>(screenData, edge, width, height);
//   vservo::addArr<<<numBlocksScreen, kNumThreads>>>(edge, prevArr, output, width, height);
//   vservo::spacing<<<numBlocksScreen, kNumThreads>>>(output, difference, count, width, height);
// }
// void VisualServo::trigger(unsigned char **imageIn, unsigned char **imageOut) {
//   // cudaMallocManaged(&matA, sizeof(unsigned char) * imageWidth * imageHeight * 3);
//   // cudaMallocManaged(&grayData, sizeof(unsigned char) * imageWidth * imageHeight);
//   // cudaMallocManaged(&screenData, sizeof(unsigned char) * width * height);
//   // cudaMallocManaged(&edge, sizeof(unsigned char) * width * height);
//   // cudaMallocManaged(&prevArr, sizeof(unsigned char) * width * height);
//   // cudaMallocManaged(&output, sizeof(unsigned char) * width * height);
//   // cudaMallocManaged(&imageInfo, sizeof(int) * 2);
//   // cudaMallocManaged(&screenInfo, sizeof(int) * 4);
//   // cudaMallocManaged(&difference, sizeof(unsigned int) * width * height);
//   // cudaMallocManaged(&count, sizeof(unsigned int) * width * height);
//   vservo::gpu_grayscale<<<numBlocksImage, kNumThreads>>>(matA, grayData, imageWidth, imageHeight);
//   vservo::screenAllocate<<<numBlocksScreen, numThreads>>>(grayData, screenData, imageInfo, screenInfo);
//   vservo::edgeFind<<<numBlocksScreen, numThreads>>>(screenData, prevArr, width, height);
//   vservo::addArr<<<numBlocksScreen, numThreads>>>(edge, prevArr, output, width, height);
//   vservo::spacing<<<numBlocksScreen, numThreads>>>(output, difference, count, width, height);
//   vservo::dilate<<<numBlocksScreen, numThreads>>>(edge, width, height);
// }
// void VisualServo::velocity(unsigned char **imageIn, unsigned char **imageOut) {
//   vservo::gpu_grayscale<<<numBlocksImage, kNumThreads>>>(matA, grayData, imageWidth, imageHeight);
//   vservo::screenAllocate<<<numBlocksScreen, kNumThreads>>>(grayData, screenData, imageInfo, screenInfo);
//   vservo::edgeFind<<<numBlocksScreen, kNumThreads>>>(screenData, edge, width, height);
//   vservo::addArr<<<numBlocksScreen, kNumThreads>>>(edge, prevArr, output, width, height);
//   vservo::spacing<<<numBlocksScreen, kNumThreads>>>(output, difference, count, width, height);
//   vservo::dilate<<<numBlocksScreen, kNumThreads>>>(edge, width, height);
// }

// double VisualServo::get_screen_velocity() {
//   for (int i = 0; i < width * height; i++) {
//     difference[0] += difference[i];
//     count[0] += count[i];
//   }
//   int currentVelocity = difference[0] / count[0];
//   int velocity = (currentVelocity + prevVelocity) / 2;
//   prevVelocity = currentVelocity;
//   printf("Velocity:%d \n", difference[0] / count[0]);
// }

void VisualServo::write_to_file(const std::string &filename, const unsigned char *lineData, int width) {
  std::ofstream dataFile;
  dataFile.open(filename);
  for (int k = 0; k < 3; k++) {
    for (int j = 100 * (k + 1) * width; j < ((100 * (k + 1)) + 1) * width; j++) {
      int output = lineData[j] - '0';
      dataFile << output;
      if (j != ((100 * (k + 1)) + 1) * width - 1) { // last element in row
        dataFile << ",";
      }
      // lineData[j] = 0;
    }
    dataFile << "\n";
  }
  dataFile.close();
}

} // namespace vservo
