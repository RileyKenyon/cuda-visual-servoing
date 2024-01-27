///-----------------------------------------------------------------------------
/// @file v4l2_sink.cpp
///
/// @brief Implementation of V4L2 video output sink
///
/// @date 2026-02-17
///-----------------------------------------------------------------------------
#include "v4l2_sink.hpp"
#include <algorithm>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <iostream>
#include <linux/videodev2.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <unistd.h>

namespace vservo {

V4L2Sink::V4L2Sink(const std::string &device_name, uint32_t width, uint32_t height)
    : device_name_(device_name), width_(width), height_(height) {
  // No initialization needed - we use direct V4L2 ioctl
}

V4L2Sink::~V4L2Sink() {
  stop();
  close_device();
}

void V4L2Sink::start() {
  if (running_) {
    throw std::runtime_error("V4L2 sink already running");
  }

  if (!open_device()) {
    throw std::runtime_error("Failed to open V4L2 device: " + device_name_);
  }

  running_ = true;
  stopped_ = false;
  write_thread_ = std::thread(&V4L2Sink::v4l2_write_thread, this);
}

void V4L2Sink::stop() {
  if (!running_) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = true;
  }
  cond_.notify_one();

  if (write_thread_.joinable()) {
    write_thread_.join();
  }

  running_ = false;
}

bool V4L2Sink::push_frame(AVFrame *frame) {
  if (!running_) {
    return false;
  }

  if (!frame) {
    return false;
  }

  // Clone the frame since we're queueing it
  AVFrame *clone = av_frame_alloc();
  if (!clone) {
    return false;
  }

  if (av_frame_ref(clone, frame) < 0) {
    av_frame_free(&clone);
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    frame_queue_.push(clone);
  }
  cond_.notify_one();

  return true;
}

bool V4L2Sink::open_device() {
  // Open the V4L2 device directly (not through FFmpeg)
  // Use O_RDWR for V4L2 devices to allow both reading device capabilities and writing frames
  int fd = ::open(device_name_.c_str(), O_RDWR);
  if (fd < 0) {
    std::cerr << "[V4L2Sink] Failed to open device " << device_name_ << " (flags=O_RDWR, errno=" << errno << ")"
              << std::endl;
    return false;
  }
  std::cout << "[V4L2Sink] Opened device: " << device_name_ << std::endl;

  // Set the video format using V4L2 ioctl
  // NOTE: We'll use conservative bytesperline since we don't know the frame linesize yet
  // This will be overridden when we actually write frames
  struct v4l2_format fmt = {};
  fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  fmt.fmt.pix.width = width_;
  fmt.fmt.pix.height = height_;
  // Use YUV420 (I420) which matches FFmpeg's AV_PIX_FMT_YUV420P format (Y,U,V order)
  // This is the most widely supported format for v4l2 loopback devices
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUV420;
  fmt.fmt.pix.field = V4L2_FIELD_NONE;
  // Don't set bytesperline - let the driver choose, or we'll handle it per-frame
  fmt.fmt.pix.bytesperline = 0;

  if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
    std::cerr << "[V4L2Sink] Failed to set format: " << strerror(errno) << std::endl;
    ::close(fd);
    return false;
  }

  // Query the format back to see what the driver actually accepted
  struct v4l2_format fmt_query = {};
  fmt_query.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  if (ioctl(fd, VIDIOC_G_FMT, &fmt_query) >= 0) {
    fmt = fmt_query; // Use the queried format
    // Log the actual pixel format (as fourcc)
    char fourcc[5] = {0};
    fourcc[0] = (fmt.fmt.pix.pixelformat) & 0xFF;
    fourcc[1] = (fmt.fmt.pix.pixelformat >> 8) & 0xFF;
    fourcc[2] = (fmt.fmt.pix.pixelformat >> 16) & 0xFF;
    fourcc[3] = (fmt.fmt.pix.pixelformat >> 24) & 0xFF;
    std::cout << "[V4L2Sink] Device accepted format: " << fourcc << " (" << fmt.fmt.pix.width << "x"
              << fmt.fmt.pix.height << ")" << std::endl;
  }

  y_bytesperline_ = fmt.fmt.pix.bytesperline;
  if (y_bytesperline_ == 0)
    y_bytesperline_ = width_;
  uv_bytesperline_ = (y_bytesperline_ + 1) / 2;

  std::cout << "[V4L2Sink] Set device format: " << fmt.fmt.pix.width << "x" << fmt.fmt.pix.height
            << " YUV420 (Y stride=" << y_bytesperline_ << " UV stride=" << uv_bytesperline_ << ")" << std::endl;

  // Store the file descriptor for writing frames
  // Note: We don't use VIDIOC_STREAMON for write()-based V4L2 devices
  device_fd_ = fd;
  device_opened_ = true;

  return true;
}

void V4L2Sink::close_device() {
  if (device_fd_ >= 0) {
    ::close(device_fd_);
    device_fd_ = -1;
  }

  device_opened_ = false;
}

bool V4L2Sink::write_frame(AVFrame *frame) {
  if (!device_opened_ || device_fd_ < 0) {
    std::cerr << "[V4L2Sink] Device not opened" << std::endl;
    return false;
  }

  if (!frame || frame->format != AV_PIX_FMT_YUV420P) {
    std::cerr << "[V4L2Sink] Frame is null or not YUV420P (format=" << (frame ? frame->format : -1) << ")" << std::endl;
    return false;
  }

  // Debug: log frame info
  static int frame_count = 0;

  // v4l2 loopback expects exactly one write() per frame (kernel treats each write as one frame).
  // Assemble the full frame in device format (YUV420: Y, U, V with device strides) then write once.
  const int h = frame->height;
  const int h2 = h / 2;
  const uint32_t y_stride = y_bytesperline_;
  const uint32_t uv_stride = uv_bytesperline_;
  const int src_y_stride = frame->linesize[0];
  const int src_u_stride = frame->linesize[1];
  const int src_v_stride = frame->linesize[2];

  const size_t frame_size = static_cast<size_t>(y_stride) * h + static_cast<size_t>(uv_stride) * h2 * 2;
  frame_buffer_.resize(frame_size);
  uint8_t *dst = frame_buffer_.data();

  // Y plane
  for (int y = 0; y < h; y++) {
    size_t copy = std::min(static_cast<size_t>(src_y_stride), static_cast<size_t>(y_stride));
    std::memcpy(dst, frame->data[0] + y * src_y_stride, copy);
    if (copy < static_cast<size_t>(y_stride))
      std::memset(dst + copy, 0, y_stride - copy);
    dst += y_stride;
  }
  // U plane
  for (int y = 0; y < h2; y++) {
    size_t copy = std::min(static_cast<size_t>(src_u_stride), static_cast<size_t>(uv_stride));
    std::memcpy(dst, frame->data[1] + y * src_u_stride, copy);
    if (copy < static_cast<size_t>(uv_stride))
      std::memset(dst + copy, 0, uv_stride - copy);
    dst += uv_stride;
  }
  // V plane
  for (int y = 0; y < h2; y++) {
    size_t copy = std::min(static_cast<size_t>(src_v_stride), static_cast<size_t>(uv_stride));
    std::memcpy(dst, frame->data[2] + y * src_v_stride, copy);
    if (copy < static_cast<size_t>(uv_stride))
      std::memset(dst + copy, 0, uv_stride - copy);
    dst += uv_stride;
  }

  ssize_t n = ::write(device_fd_, frame_buffer_.data(), frame_size);
  if (n < 0) {
    std::cerr << "[V4L2Sink] Error writing frame: " << strerror(errno) << std::endl;
    return false;
  }
  if (static_cast<size_t>(n) != frame_size) {
    std::cerr << "[V4L2Sink] Short write: " << n << " != " << frame_size << std::endl;
    return false;
  }
  frame_count++;

  return true;
}

void V4L2Sink::v4l2_write_thread() {
  static int thread_frame_count = 0;
  std::cout << "[V4L2Sink] Write thread started" << std::endl;

  while (true) {
    AVFrame *frame = nullptr;

    {
      std::unique_lock<std::mutex> lock(mutex_);
      cond_.wait(lock, [this]() { return !frame_queue_.empty() || stopped_; });

      if (stopped_ && frame_queue_.empty()) {
        std::cout << "[V4L2Sink] Write thread stopping, processed " << thread_frame_count << " frames" << std::endl;
        break;
      }

      if (!frame_queue_.empty()) {
        frame = frame_queue_.front();
        frame_queue_.pop();
      }
    }

    if (frame) {
      if (!write_frame(frame)) {
        std::cerr << "[V4L2Sink] Error writing frame " << thread_frame_count << std::endl;
      }
      av_frame_free(&frame);
      thread_frame_count++;
    }
  }
}

} // namespace vservo
