///-----------------------------------------------------------------------------
/// @file scrcpy_sink.hpp
///
/// @author Riley Kenyon (rike2277@colorado.edu)
/// @brief Custom scrcpy client using POSIX sockets for video and control
///
/// Implements scrcpy protocol over raw sockets (H.264 video + control messages).
/// Decodes video to V4L2 device output.
///
/// @date 2026-02-17
///-----------------------------------------------------------------------------

#ifndef SCRCPY_SINK_HPP
#define SCRCPY_SINK_HPP

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
}

#include "v4l2_sink.hpp"

namespace vservo {

/// Represents a video frame packet from H.264 stream
struct VideoFramePacket {
  uint64_t pts;              ///< Presentation timestamp (microseconds)
  bool key_frame;            ///< Whether this is a keyframe
  bool config_packet;        ///< Whether this is config (SPS/PPS)
  std::vector<uint8_t> data; ///< Raw H.264 data
};

/// Video socket - reads H.264 stream from device
class VideoSocket {
public:
  /// Construct with port to listen on
  explicit VideoSocket(int port);
  ~VideoSocket();

  /// Accept incoming connection from device
  /// @return true on success, false on error
  bool accept_connection();

  /// Read raw bytes from socket (blocking)
  /// Reads exactly N bytes or returns false on error
  /// @param buffer destination buffer
  /// @param size number of bytes to read
  /// @return true if exactly size bytes read, false otherwise
  bool read_bytes(uint8_t *buffer, size_t size);

  /// Close the socket
  void close();

  /// Check if connection is active
  bool is_connected() const { return client_fd_ >= 0; }

private:
  int client_fd_{-1};     ///< Connected client socket
  bool connected_{false}; ///< True once connected via connect()
  int port_;
};

/// Control socket - sends control messages to device
class ControlSocket {
public:
  /// Construct with port to listen on
  explicit ControlSocket(int port);
  ~ControlSocket();

  /// Accept incoming connection from device
  /// @return true on success, false on error
  bool accept_connection();

  /// Send raw bytes to device (blocking)
  /// @param buffer data to send
  /// @param size number of bytes
  /// @return true if all bytes sent, false otherwise
  bool write_bytes(const uint8_t *buffer, size_t size);

  /// Close the socket
  void close();

  /// Check if connection is active
  bool is_connected() const { return client_fd_ >= 0; }

private:
  int client_fd_{-1};     ///< Connected client socket
  bool connected_{false}; ///< True once connected via connect()
  int port_;
};

/// Manages H.264 frame decoding
class H264Decoder {
public:
  H264Decoder();
  ~H264Decoder();

  bool init(uint32_t width, uint32_t height);
  // Set codec extradata (e.g., avcC / SPS+PPS) before opening decoder
  bool set_extradata(const uint8_t *data, size_t size);
  // Open codec after extradata and dimensions are set
  bool open_codec();
  AVFrame *decode_packet(const uint8_t *packet_data, size_t packet_size, uint64_t pts);

  /// Flush remaining frames from decoder
  /// @return last available AVFrame or nullptr
  AVFrame *flush();

private:
  AVCodecContext *codec_ctx_{nullptr};
  AVFrame *frame_{nullptr};
  AVPacket *packet_{nullptr};
};

/// Main scrcpy sink - coordinates sockets, decoding, and V4L2 output
class ScrcpySink {
public:
  /// Construct sink
  /// @param v4l2_device V4L2 device path (e.g., "/dev/video0")
  /// @param port port to listen for scrcpy server connection (default 27183)
  explicit ScrcpySink(const std::string &v4l2_device, int port = 1234);

  /// Destructor
  ~ScrcpySink();

  /// Start receiving video and control
  /// Blocks until sockets connect and starts background threads
  /// @return true on success
  bool start();

  /// Stop sink gracefully
  void stop();

  /// Check if sink is running
  bool is_running() const { return running_; }

  /// Send touch event
  /// @param pointer_id unique touch pointer ID
  /// @param x screen x coordinate (device pixels)
  /// @param y screen y coordinate (device pixels)
  /// @param size contact size
  /// @param pressure pressure value (0-1000)
  void send_touch_event(uint8_t action,
                        uint64_t pointer_id,
                        uint32_t x,
                        uint32_t y,
                        uint16_t screen_width,
                        uint16_t screen_height,
                        float pressure,
                        uint32_t action_button,
                        uint32_t buttons);

  /// Get current video width
  uint32_t get_width() const { return video_width_; }

  /// Get current video height
  uint32_t get_height() const { return video_height_; }

  // TODO: Add frame dumping support
  // /// Enable debug frame dumping to file
  // void enable_frame_dump(const std::string &output_file);
  // void disable_frame_dump();

  // Backward compatibility methods

  /// Cleanup and stop (alias for stop())
  void cleanup() { stop(); }

  /// @brief Touch the screen
  /// @param x x position in pixels
  /// @param y y position in pixels
  void touch(std::size_t x, std::size_t y);

private:
  /// Thread function - reads video frames and decodes them
  void video_reader_thread();

  /// Thread function - manages control socket (currently just listens for device messages)
  void control_reader_thread();

  /// Send a complete control message
  bool send_control_message(const std::vector<uint8_t> &message);

  // Configuration
  std::string v4l2_device_;
  int port_;

  // Socket management
  std::unique_ptr<VideoSocket> video_socket_;
  std::unique_ptr<ControlSocket> control_socket_;
  std::unique_ptr<H264Decoder> decoder_;
  std::unique_ptr<V4L2Sink> v4l2_sink_;

  // Video parameters from codec metadata
  uint32_t video_width_{0};
  uint32_t video_height_{0};
  // Device info (first socket sends device name)
  bool device_info_read_{false};
  std::string device_name_;

  // Thread management
  std::thread video_thread_;
  std::thread control_thread_;
  std::mutex state_mutex_;
  std::condition_variable state_cond_;
  bool running_{false};
  bool stopped_{false};
  bool sockets_connected_{false};
};

} // namespace vservo

#endif // SCRCPY_SINK_HPP
