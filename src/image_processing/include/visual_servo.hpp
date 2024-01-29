///-----------------------------------------------------------------------------
/// @file visual_servo.hpp
///
/// @author Riley Kenyon (rike2277@colorado.edu)
/// @brief Control object for visual servoing
///
/// @date 2024-01-28
///-----------------------------------------------------------------------------
#ifndef VISUAL_SERVO_HPP
#define VISUAL_SERVO_HPP
#include <cuda_runtime_api.h>

namespace vservo {
class VisualServo {
public:
  explicit VisualServo(unsigned int width, unsigned int height, unsigned char layers);
  ~VisualServo();
  void report_fps(bool shouldReport);
  bool process(unsigned char *data);
  void get_output(unsigned char **data);
  void set_threads(unsigned int threadCount);

private:
  void get_framerate(); // helper to report on framerate
  unsigned int numThreads = 1024;
  unsigned int numBlocks;
  unsigned int imageWidth;
  unsigned int imageHeight;
  unsigned int imageLayers;
  unsigned long counter;
  unsigned int frameWindow = 120; ///< Number of frames used in reporting window
  bool reportFramerate = true;
  float timer;
  unsigned char *input;
  unsigned char *grayscale;
  unsigned char *output;
  cudaEvent_t start;
  cudaEvent_t stop;
};

} // namespace vservo
#endif // VISUAL_SERVO_HPP
