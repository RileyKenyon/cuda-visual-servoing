///-----------------------------------------------------------------------------
/// @file main.cpp
///
/// @author Riley Kenyon (rike2277@colorado.edu)
/// @brief Main application for visual servoing
///
/// @date 2024-01-28
///-----------------------------------------------------------------------------
#include "visual_servo.hpp"
#include <memory>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <unistd.h>

static constexpr unsigned int kNumThreads = 1024;

int main(int argc, char *argv[]) {
  bool cropImage = false;
  bool displayStream = false;
  std::string inputFilename;
  std::string outputFilename;

  int opterr = 0;
  int c;

  while ((c = getopt(argc, argv, "cdi:o:")) != -1) {
    switch (c) {
    case 'c':
      cropImage = true;
      break;
    case 'd':
      displayStream = true;
      break;
    case 'i':
      inputFilename = optarg;
      break;
    case 'o':
      outputFilename = optarg;
      break;
    case '?':
      if (optopt == 'i' || optopt == 'o')
        fprintf(stderr, "Option -%c requires an argument.\n", optopt);
      else if (isprint(optopt))
        fprintf(stderr, "Unknown option `-%c'.\n", optopt);
      else
        fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
      return 1;
    default:
      abort();
    }
  }

  printf("crop = %u, display = %u, input = %s, output = %u\n",
         cropImage,
         displayStream,
         inputFilename.c_str(),
         !outputFilename.empty());

  for (int index = optind; index < argc; index++) {
    printf("Non-option argument %s\n", argv[index]);
  }

  // Initialize capture source
  cv::Mat img;
  cv::VideoCapture cap = (inputFilename.empty()) ? cv::VideoCapture(1) : cv::VideoCapture(inputFilename);
  if (!cap.isOpened()) {
    std::runtime_error("Error getting Stream");
  }
  int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

  // Configure video writer
  std::unique_ptr<cv::VideoWriter> writer = nullptr;
  if (!outputFilename.empty()) {
    double fps = cap.get(cv::CAP_PROP_FPS);
    bool useColor = true;
    writer = std::make_unique<cv::VideoWriter>();
    if (cap.get(cv::CAP_PROP_FRAME_COUNT) == 1) {
      auto codec = cap.get(cv::CAP_PROP_FOURCC);
      writer->open(outputFilename, cv::CAP_IMAGES, codec, 1, cv::Size(width, height), useColor);
    } else {
      int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
      writer->open(outputFilename, codec, fps, cv::Size(width, height), useColor);
    }
  }

  // Show image if configured
  if (displayStream) {
    // Crop image if configured
    if (cropImage && cap.read(img)) {
      cv::Rect2d r = cv::selectROI(img);
      width = r.width;
      height = r.height;
    }
  }

  // Setup visual servo
  vservo::VisualServo vs(width, height, 3);
  vs.set_threads(kNumThreads);
  vs.report_fps(true);

  while (true) {
    // read
    if (!cap.read(img)) {
      break;
    }

    // processs
    vs.process(img.data);
    unsigned char *output;
    vs.get_output(&output);

    // visualize
    if (displayStream) {
      cv::imshow("Preview", img);
      c = cv::waitKey(1);
      if (c == 'p') {
        cv::imwrite("Preview_%02d.jpg", img);
      } else if (c == 'q') {
        break;
      }
    }

    // write to file
    if (nullptr != writer && nullptr != output) {
      cv::Mat outputGrayscale(cv::Size(width, height), CV_8UC1, output);
      cv::Mat outputColor;
      cv::cvtColor(outputGrayscale, outputColor, cv::COLOR_GRAY2BGR);
      writer->write(outputColor);
    }
  }
  cap.release();
  return 0;
}
