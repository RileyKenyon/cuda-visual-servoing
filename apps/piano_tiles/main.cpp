///-----------------------------------------------------------------------------
/// @file main.cpp
///
/// @author Riley Kenyon (rike2277@colorado.edu)
/// @brief Main application for visual servoing
///
/// @date 2024-01-28
///-----------------------------------------------------------------------------
#include <csignal>
#include <iostream>
#include <memory>
#include <thread>
#include <unistd.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include "scrcpy_sink.hpp"
#include "visual_servo.hpp"

static constexpr char device_name[] = "/dev/video0";
std::unique_ptr<vservo::ScrcpySink> sink = nullptr;

/// Signignal handler for SIGINT
void signal_handler(int signal) {
  if (signal == SIGINT) {
    std::cout << "SIGINT received, shutting down..." << std::endl;
    if (sink != nullptr) {
      sink->cleanup();
    }
    exit(0);
  }
}

static constexpr unsigned int kNumThreads = 1024;

int main(int argc, char *argv[]) {
  struct sigaction sa;
  sa.sa_handler = signal_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  sigaction(SIGINT, &sa, nullptr);

  std::string inputFilename;
  std::string outputFilename;

  int c;
  while ((c = getopt(argc, argv, "i:o:")) != -1) {
    switch (c) {
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

  // Create the sink
  sink = std::make_unique<vservo::ScrcpySink>(device_name);
  sink->start();

  // Wait for the device to get set up
  std::this_thread::sleep_for(std::chrono::milliseconds(5000));

  // Initialize capture source
  const bool fromV4L2Device = inputFilename.empty();
  cv::Mat img;
  cv::VideoCapture cap = fromV4L2Device ? cv::VideoCapture(device_name, cv::CAP_V4L2) : cv::VideoCapture(inputFilename);
  if (!cap.isOpened()) {
    std::runtime_error("Error getting Stream");
  }
  // Only request MJPEG when reading from file; V4L2 device outputs YUV420 natively
  if (!fromV4L2Device)
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
  int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  std::cout << "Stream opened: " << width << "x" << height << (fromV4L2Device ? " (V4L2)" : "") << std::endl;

  // Setup visual servo (use initial size; may update when first frame is read)
  vservo::VisualServo vs(width > 0 ? width : 1080, height > 0 ? height : 1920, 3);
  vs.set_threads(kNumThreads);
  vs.report_fps(true);

  // Writer opened after first frame so we use actual dimensions
  std::unique_ptr<cv::VideoWriter> writer = nullptr;
  if (!outputFilename.empty()) {
    writer = std::make_unique<cv::VideoWriter>();
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    if (writer->open(outputFilename,
                     cv::CAP_FFMPEG,
                     codec,
                     cap.get(cv::CAP_PROP_FPS) > 0.0 ? cap.get(cv::CAP_PROP_FPS) : 30.0,
                     cv::Size(sink->get_width(), sink->get_height()),
                     true)) {
      std::cout << "Recording to " << outputFilename << " " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
                << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << " @" << cap.get(cv::CAP_PROP_FPS) << " fps" << std::endl;
    } else {
      std::cerr << "Failed to open VideoWriter for " << outputFilename << std::endl;
      writer.reset();
    }
  }

  for (std::uint32_t i = 0; i < 400; ++i) {
    if (!cap.read(img)) {
      std::cout << "Unable to read image" << std::endl;
      break;
    }

    auto point = vs.process_frame(img);

    if (point) {
      sink->touch(point->x, point->y);
    }

    if (writer) {
      cv::Mat frame;
      if (img.cols != sink->get_width() || img.rows != sink->get_height()) {
        cv::resize(img, frame, cv::Size(sink->get_width(), sink->get_height()));
      } else if (!img.isContinuous())
        frame = img.clone();
      else
        frame = img;
      if (point) {
        cv::circle(frame, *point, 24, cv::Scalar(0, 255, 0), 2);
        cv::circle(frame, *point, 4, cv::Scalar(0, 255, 0), -1);
      }
      writer->write(frame);
    }
  }
  cap.release();
  return 0;
}
