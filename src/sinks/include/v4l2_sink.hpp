///-----------------------------------------------------------------------------
/// @file v4l2_sink.hpp
///
/// @brief V4L2 video output sink for decoded frames
///
/// Handles H.264 frame buffering, decoding to raw AVFrame, and output to
/// V4L2 device using FFmpeg's v4l2 muxer.
///
/// @date 2026-02-17
///-----------------------------------------------------------------------------
#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
}

namespace vservo {

/// Represents a single video frame packet
struct VideoPacket {
  uint64_t pts;              ///< Presentation timestamp (in microseconds)
  bool key_frame;            ///< Whether this is a keyframe
  bool config_packet;        ///< Whether this is a config/SPS/PPS packet
  std::vector<uint8_t> data; ///< Raw H.264 packet data
};

/// V4L2 video output sink
/// Decodes H.264 packets and writes decoded frames to V4L2 device
class V4L2Sink {
public:
  /// Initialize sink with device path and video parameters
  /// @param device_name V4L2 device path (e.g., "/dev/video0")
  /// @param width Initial video width
  /// @param height Initial video height
  V4L2Sink(const std::string &device_name, uint32_t width, uint32_t height);

  ~V4L2Sink();

  /// Start the V4L2 output thread
  /// Must be called before pushing frames
  void start();

  /// Stop the V4L2 output thread
  void stop();

  /// Push a decoded frame to the V4L2 sink
  /// @param frame AVFrame with YUV420P pixel format
  /// @return true if successful, false on error
  bool push_frame(AVFrame *frame);

  /// Get the current video width
  uint32_t get_width() const { return width_; }

  /// Get the current video height
  uint32_t get_height() const { return height_; }

  /// Check if sink is running
  bool is_running() const { return running_; }

private:
  /// Thread function for V4L2 writing
  void v4l2_write_thread();

  /// Open the V4L2 device and configure encoder
  bool open_device();

  /// Close the V4L2 device
  void close_device();

  /// Write AVFrame to V4L2 device
  bool write_frame(AVFrame *frame);

  // Configuration
  std::string device_name_;
  uint32_t width_;
  uint32_t height_;

  // Device stride (from VIDIOC_S_FMT); must match when writing
  uint32_t y_bytesperline_{0};
  uint32_t uv_bytesperline_{0};

  /// Reusable buffer for one frame (YUV420: Y then U then V, device strides)
  /// v4l2 loopback expects exactly one write() per frame.
  std::vector<uint8_t> frame_buffer_;

  // V4L2 device file descriptor
  int device_fd_{-1};

  // Thread synchronization
  std::thread write_thread_;
  std::mutex mutex_;
  std::condition_variable cond_;
  std::queue<AVFrame *> frame_queue_;
  bool running_{false};
  bool stopped_{false};
  bool header_written_{false};

  // State
  bool device_opened_{false};
};

} // namespace vservo
