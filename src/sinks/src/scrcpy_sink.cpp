#include "scrcpy_sink.hpp"

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace vservo {

// ============================================================================
// VideoSocket Implementation
// ============================================================================

VideoSocket::VideoSocket(int port) : port_(port) {
  // Create client socket (we connect to device/localhost)
  client_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (client_fd_ < 0) {
    throw std::runtime_error("Failed to create video client socket: " + std::string(strerror(errno)));
  }

  // Non-blocking connect will be performed in accept_connection (which acts as connect)
}

VideoSocket::~VideoSocket() { close(); }

bool VideoSocket::accept_connection() {
  const char *host = "127.0.0.1";
  unsigned attempts = 100;
  int delay_ms = 100;

  for (unsigned i = 0; i < attempts; ++i) {
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    inet_pton(AF_INET, host, &addr.sin_addr);

    if (::connect(client_fd_, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
      // FIRST socket: read dummy byte to confirm server is ready
      uint8_t b;
      ssize_t n = ::recv(client_fd_, &b, 1, 0);
      if (n == 1) {
        connected_ = true;
        std::cout << "[VideoSocket] Connected and initial byte received" << std::endl;
        return true;
      }
      if (n == 0) {
        std::cerr << "[VideoSocket] Connection closed immediately after connect" << std::endl;
      } else {
        std::cerr << "[VideoSocket] No initial byte received (err=" << strerror(errno) << ")" << std::endl;
      }
      ::close(client_fd_);
      client_fd_ = socket(AF_INET, SOCK_STREAM, 0);
      connected_ = false;
      if (client_fd_ < 0) {
        std::cerr << "[VideoSocket] Failed to recreate socket: " << strerror(errno) << std::endl;
        return false;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
  }
  std::cerr << "[VideoSocket] Could not connect to " << host << ":" << port_ << " after attempts" << std::endl;
  return false;
}

bool VideoSocket::read_bytes(uint8_t *buffer, size_t size) {
  if (!connected_ || client_fd_ < 0) {
    return false;
  }

  size_t bytes_read = 0;
  while (bytes_read < size) {
    ssize_t n = ::read(client_fd_, buffer + bytes_read, size - bytes_read);
    if (n < 0) {
      std::cerr << "[VideoSocket] Read error: " << strerror(errno) << std::endl;
      return false;
    }
    if (n == 0) {
      // Connection closed by peer
      std::cerr << "[VideoSocket] Connection closed by device" << std::endl;
      return false;
    }
    bytes_read += n;
  }

  return true;
}

void VideoSocket::close() {
  if (client_fd_ >= 0) {
    ::close(client_fd_);
    client_fd_ = -1;
    connected_ = false;
  }
}

// ============================================================================
// ControlSocket Implementation
// ============================================================================

ControlSocket::ControlSocket(int port) : port_(port) {
  // Control client socket connects to port on device/localhost
  client_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (client_fd_ < 0) {
    throw std::runtime_error("Failed to create control client socket: " + std::string(strerror(errno)));
  }
  // Disable Nagle's algorithm for low-latency control messages
  int flag = 1;
  if (setsockopt(client_fd_, IPPROTO_TCP, O_NDELAY, &flag, sizeof(flag)) < 0) {
    std::cerr << "[ControlSocket] Failed to disable Nagle's algorithm: " << strerror(errno) << std::endl;
  }
}

ControlSocket::~ControlSocket() { close(); }

bool ControlSocket::accept_connection() {
  const char *host = "127.0.0.1";
  unsigned attempts = 100;
  int delay_ms = 100;

  for (unsigned i = 0; i < attempts; ++i) {
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    inet_pton(AF_INET, host, &addr.sin_addr);

    if (::connect(client_fd_, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
      // SECOND socket: do NOT read dummy byte, just mark as connected
      connected_ = true;
      std::cout << "[ControlSocket] Connected (no dummy byte)" << std::endl;
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
  }
  std::cerr << "[ControlSocket] Could not connect to " << host << ":" << (port_) << " after attempts" << std::endl;
  return false;
}

bool ControlSocket::write_bytes(const uint8_t *buffer, size_t size) {
  if (!connected_ || client_fd_ < 0) {
    return false;
  }

  size_t bytes_written = 0;
  while (bytes_written < size) {
    ssize_t n = ::write(client_fd_, buffer + bytes_written, size - bytes_written);
    if (n < 0) {
      std::cerr << "[ControlSocket] Write error: " << strerror(errno) << std::endl;
      return false;
    }
    if (n == 0) {
      std::cerr << "[ControlSocket] Could not write bytes" << std::endl;
      return false;
    }
    bytes_written += n;
  }

  return true;
}

void ControlSocket::close() {
  if (client_fd_ >= 0) {
    ::close(client_fd_);
    client_fd_ = -1;
    connected_ = false;
  }
}

// ============================================================================
// H264Decoder Implementation
// ============================================================================

H264Decoder::H264Decoder() {
  const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_H264);
  if (!codec) {
    throw std::runtime_error("H.264 codec not found");
  }

  codec_ctx_ = avcodec_alloc_context3(codec);
  if (!codec_ctx_) {
    throw std::runtime_error("Failed to allocate codec context");
  }

  frame_ = av_frame_alloc();
  packet_ = av_packet_alloc();

  if (!frame_ || !packet_) {
    throw std::runtime_error("Failed to allocate frame or packet");
  }
}

H264Decoder::~H264Decoder() {
  if (codec_ctx_) {
    avcodec_free_context(&codec_ctx_);
  }
  if (frame_) {
    av_frame_free(&frame_);
  }
  if (packet_) {
    av_packet_free(&packet_);
  }
}

bool H264Decoder::init(uint32_t width, uint32_t height) {
  if (!codec_ctx_) {
    return false;
  }

  codec_ctx_->width = width;
  codec_ctx_->height = height;
  codec_ctx_->pix_fmt = AV_PIX_FMT_YUV420P;

  // Do not open codec here if extradata is required - caller should invoke open_codec()
  std::cout << "[H264Decoder] Dimensions set for " << width << "x" << height << std::endl;
  return true;
}

bool H264Decoder::set_extradata(const uint8_t *data, size_t size) {
  if (!codec_ctx_)
    return false;
  // Free existing extradata if present
  if (codec_ctx_->extradata) {
    av_free(codec_ctx_->extradata);
    codec_ctx_->extradata = nullptr;
    codec_ctx_->extradata_size = 0;
  }

  // Allocate and copy extradata (with padding)
  uint8_t *e = (uint8_t *)av_malloc(size + AV_INPUT_BUFFER_PADDING_SIZE);
  if (!e)
    return false;
  memcpy(e, data, size);
  // Zero padding
  memset(e + size, 0, AV_INPUT_BUFFER_PADDING_SIZE);
  codec_ctx_->extradata = e;
  codec_ctx_->extradata_size = (int)size;
  std::cout << "[H264Decoder] Extradata set (" << size << " bytes)" << std::endl;
  return true;
}

bool H264Decoder::open_codec() {
  if (!codec_ctx_) {
    std::cerr << "[H264Decoder] open_codec: codec_ctx is null" << std::endl;
    return false;
  }

  if (!codec_ctx_->codec) {
    std::cerr << "[H264Decoder] open_codec: codec not set" << std::endl;
    return false;
  }

  std::cout << "[H264Decoder] Opening codec with:"
            << " width=" << codec_ctx_->width << " height=" << codec_ctx_->height << " pix_fmt=" << codec_ctx_->pix_fmt
            << " extradata_size=" << codec_ctx_->extradata_size << std::endl;

  if (avcodec_open2(codec_ctx_, codec_ctx_->codec, nullptr) < 0) {
    std::cerr << "[H264Decoder] Failed to open H.264 decoder" << std::endl;
    return false;
  }
  std::cout << "[H264Decoder] Opened codec successfully for " << codec_ctx_->width << "x" << codec_ctx_->height
            << std::endl;
  return true;
}

AVFrame *H264Decoder::decode_packet(const uint8_t *packet_data, size_t packet_size, uint64_t pts) {
  if (!codec_ctx_ || !packet_ || !frame_) {
    std::cerr << "[H264Decoder] decode_packet: codec_ctx=" << (codec_ctx_ ? "yes" : "no")
              << ", packet=" << (packet_ ? "yes" : "no") << ", frame=" << (frame_ ? "yes" : "no") << std::endl;
    return nullptr;
  }

  // Set packet data
  packet_->data = const_cast<uint8_t *>(packet_data);
  packet_->size = packet_size;
  packet_->pts = pts;

  int ret = avcodec_send_packet(codec_ctx_, packet_);
  if (ret < 0 && ret != AVERROR(EAGAIN)) {
    std::cerr << "[H264Decoder] Failed to send packet (size=" << packet_size << "): " << ret << std::endl;
    return nullptr;
  }

  ret = avcodec_receive_frame(codec_ctx_, frame_);
  if (ret == 0) {
    // Frame decoded successfully - create a copy to return
    AVFrame *out_frame = av_frame_alloc();
    if (!out_frame || av_frame_ref(out_frame, frame_) < 0) {
      std::cerr << "[H264Decoder] Failed to ref frame" << std::endl;
      av_frame_free(&out_frame);
      return nullptr;
    }
    av_frame_unref(frame_);
    return out_frame;
  } else if (ret == AVERROR(EAGAIN)) {
    // Need more data
    return nullptr;
  } else {
    std::cerr << "[H264Decoder] Failed to receive frame (ret=" << ret << ")" << std::endl;
    return nullptr;
  }
}

AVFrame *H264Decoder::flush() {
  if (!codec_ctx_ || !packet_) {
    return nullptr;
  }

  // Send null packet to flush
  int ret = avcodec_send_packet(codec_ctx_, nullptr);
  if (ret < 0) {
    return nullptr;
  }

  ret = avcodec_receive_frame(codec_ctx_, frame_);
  if (ret == 0) {
    AVFrame *out_frame = av_frame_alloc();
    if (!out_frame || av_frame_ref(out_frame, frame_) < 0) {
      av_frame_free(&out_frame);
      return nullptr;
    }
    av_frame_unref(frame_);
    return out_frame;
  }

  return nullptr;
}

// ============================================================================
// ScrcpySink Implementation
// ============================================================================

ScrcpySink::ScrcpySink(const std::string &v4l2_device, int port) : v4l2_device_(v4l2_device), port_(port) {

  // av_register_all and avcodec_register_all are no longer needed in newer FFmpeg
  // avdevice_register_all(); // Will be called in start()

  std::cout << "[ScrcpySink] Created with V4L2 device: " << v4l2_device_ << ", port: " << port_ << std::endl;
}

ScrcpySink::~ScrcpySink() { stop(); }

bool ScrcpySink::start() {
  if (running_) {
    std::cerr << "[ScrcpySink] Already running" << std::endl;
    return false;
  }

  try {
    // Create sockets
    video_socket_ = std::make_unique<VideoSocket>(port_);
    control_socket_ = std::make_unique<ControlSocket>(port_);
    decoder_ = std::make_unique<H264Decoder>();

    // Start video reader thread
    std::cout << "[ScrcpySink] Starting video reader thread..." << std::endl;
    video_thread_ = std::thread(&ScrcpySink::video_reader_thread, this);

    // Start control reader thread
    std::cout << "[ScrcpySink] Starting control reader thread..." << std::endl;
    control_thread_ = std::thread(&ScrcpySink::control_reader_thread, this);

    running_ = true;
    std::cout << "[ScrcpySink] Started successfully" << std::endl;
    return true;

  } catch (const std::exception &e) {
    std::cerr << "[ScrcpySink] Failed to start: " << e.what() << std::endl;
    return false;
  }
}

void ScrcpySink::stop() {
  if (!running_) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    stopped_ = true;
  }
  state_cond_.notify_all();

  if (video_thread_.joinable()) {
    video_thread_.join();
  }
  if (control_thread_.joinable()) {
    control_thread_.join();
  }

  if (v4l2_sink_) {
    v4l2_sink_->stop();
  }

  video_socket_.reset();
  control_socket_.reset();
  decoder_.reset();

  running_ = false;
  std::cout << "[ScrcpySink] Stopped" << std::endl;
}

void ScrcpySink::video_reader_thread() {
  std::cout << "[VideoReader] Thread started, waiting for connection..." << std::endl;

  // Wait for video socket to connect
  if (!video_socket_->accept_connection()) {
    std::cerr << "[VideoReader] Failed to accept connection" << std::endl;
    return;
  }

  // The first socket that connects must read the device name (64 bytes)
  {
    std::unique_lock<std::mutex> lock(state_mutex_);
    if (!device_info_read_) {
      uint8_t namebuf[64];
      // Use the video socket to read device name
      if (!video_socket_->read_bytes(namebuf, sizeof(namebuf))) {
        std::cerr << "[VideoReader] Failed to read device name" << std::endl;
        return;
      }
      // Ensure null-terminated
      namebuf[63] = '\0';
      device_name_ = std::string(reinterpret_cast<char *>(namebuf));
      device_info_read_ = true;
      state_cond_.notify_all();
      std::cout << "[VideoReader] Device name: " << device_name_ << std::endl;
    }
  }

  // Phase 1: Read codec metadata (12 bytes)
  uint8_t metadata[12];
  if (!video_socket_->read_bytes(metadata, 12)) {
    std::cerr << "[VideoReader] Failed to read codec metadata" << std::endl;
    return;
  }

  // Parse metadata
  // #define SC_CODEC_ID_H264 UINT32_C(0x68323634) // "h264" in ASCII
  uint32_t codec_id = (metadata[0] << 24) | (metadata[1] << 16) | (metadata[2] << 8) | metadata[3];
  uint32_t width = (metadata[4] << 24) | (metadata[5] << 16) | (metadata[6] << 8) | metadata[7];
  uint32_t height = (metadata[8] << 24) | (metadata[9] << 16) | (metadata[10] << 8) | metadata[11];

  std::cout << "[VideoReader] Codec metadata: codec_id=" << codec_id << ", width=" << width << ", height=" << height
            << std::endl;

  video_width_ = width;
  video_height_ = height;

  // Initialize decoder dimensions (codec will be opened after config is applied)
  if (!decoder_->init(width, height)) {
    std::cerr << "[VideoReader] Failed to initialize decoder" << std::endl;
    return;
  }

  bool decoder_opened = false;

  // Initialize V4L2 sink
  try {
    v4l2_sink_ = std::make_unique<V4L2Sink>(v4l2_device_, width, height);
    v4l2_sink_->start();
  } catch (const std::exception &e) {
    std::cerr << "[VideoReader] Failed to initialize V4L2 sink: " << e.what() << std::endl;
    return;
  }

  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    sockets_connected_ = true;
  }
  state_cond_.notify_all();

  std::cout << "[VideoReader] Ready to receive video frames" << std::endl;

  // Phase 2: Read frame headers and packets
  uint8_t frame_header[12];
  uint64_t frame_count = 0;

  while (!stopped_) {
    // Read frame header (12 bytes: flags+PTS in 8 bytes + packet_size in 4 bytes)
    if (!video_socket_->read_bytes(frame_header, 12)) {
      std::cout << "[VideoReader] Connection closed after " << frame_count << " frames" << std::endl;
      break;
    }

    // Parse frame header
    // Bytes 0-7: config_packet (bit 63) | key_frame (bit 62) | PTS (62 bits)
    uint64_t header_val = 0;
    for (int i = 0; i < 8; i++) {
      header_val = (header_val << 8) | frame_header[i];
    }

    bool config_packet = (header_val & 0x8000000000000000ULL) != 0;
    bool key_frame = (header_val & 0x4000000000000000ULL) != 0;
    uint64_t pts = header_val & 0x3FFFFFFFFFFFFFFFULL;

    // Bytes 8-11: packet size
    uint32_t packet_size =
        (frame_header[8] << 24) | (frame_header[9] << 16) | (frame_header[10] << 8) | frame_header[11];

    // Read packet data
    std::vector<uint8_t> packet_data(packet_size);
    if (!video_socket_->read_bytes(packet_data.data(), packet_size)) {
      std::cerr << "[VideoReader] Failed to read packet data" << std::endl;
      break;
    }

    // If this is a config packet, use it to set extradata before opening the codec
    if (config_packet && !decoder_opened) {
      if (!decoder_->set_extradata(packet_data.data(), packet_size)) {
        std::cerr << "[VideoReader] Failed to set decoder extradata" << std::endl;
      } else {
        if (!decoder_->open_codec()) {
          std::cerr << "[VideoReader] Failed to open decoder after setting extradata" << std::endl;
        } else {
          decoder_opened = true;
        }
      }
      // Config packet does not produce a decoded frame by itself; continue
      frame_count++;
      continue;
    }

    if (!decoder_opened) {
      // No config packet received yet; try opening codec without extradata
      if (!decoder_->open_codec()) {
        std::cerr << "[VideoReader] Failed to open decoder" << std::endl;
        break;
      }
      decoder_opened = true;
    }

    // Decode packet
    AVFrame *decoded_frame = decoder_->decode_packet(packet_data.data(), packet_size, pts);
    if (decoded_frame) {
      // Check if frame data is non-zero
      uint8_t *y_data = decoded_frame->data[0];
      uint32_t y_sum = 0;
      if (y_data && decoded_frame->linesize[0] > 0) {
        for (int i = 0; i < std::min(1024, (int)(decoded_frame->linesize[0] * 10)); i++) {
          y_sum += y_data[i];
        }
      } else {
        std::cout << "[VideoReader] WARNING: Frame has no Y plane data or invalid linesize" << std::endl;
      }

      // Push to V4L2 sink
      if (!v4l2_sink_->push_frame(decoded_frame)) {
        std::cerr << "[VideoReader] Failed to push frame to V4L2" << std::endl;
      }
      // TODO: Add frame dumper support later
      // if (frame_dumper_) {
      //     frame_dumper_->push_frame(decoded_frame);
      // }
      av_frame_free(&decoded_frame);
    } else {
      if (frame_count < 5 || frame_count % 30 == 0) {
        std::cout << "[VideoReader] Frame " << frame_count << " not decoded (need more data?)" << std::endl;
      }
    }

    frame_count++;
  }

  std::cout << "[VideoReader] Thread ending" << std::endl;
}

void ScrcpySink::control_reader_thread() {
  std::cout << "[ControlReader] Thread started, waiting for connection..." << std::endl;

  // Wait for control socket to connect
  if (!control_socket_->accept_connection()) {
    std::cerr << "[ControlReader] Failed to accept connection" << std::endl;
    return;
  }

  std::cout << "[ControlReader] Ready to receive control messages" << std::endl;

  // For now, just read and discard device messages (clipboard, etc.)
  uint8_t buffer[1024];
  while (!stopped_) {
    // Non-blocking read attempt - read if data available
    // For Phase 1 prove-out, we don't process device messages yet
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  std::cout << "[ControlReader] Thread ending" << std::endl;
}

bool ScrcpySink::send_control_message(const std::vector<uint8_t> &message) {
  if (!control_socket_ || !control_socket_->is_connected()) {
    std::cerr << "[ScrcpySink] Control socket not connected" << std::endl;
    return false;
  }

  return control_socket_->write_bytes(message.data(), message.size());
}

void ScrcpySink::touch(std::size_t x, std::size_t y) {

  uint8_t action = 0; // DOWN
  uint64_t pointer_id = UINT64_C(-2);
  float pressure = 1.0f;
  uint32_t action_button = 1;
  uint32_t buttons = 1;
  send_touch_event(action, pointer_id, x, y, get_width(), get_height(), pressure, action_button, buttons);
  action = 1; // UP
  send_touch_event(action, pointer_id, x, y, get_width(), get_height(), pressure, action_button, buttons);
}

void ScrcpySink::send_touch_event(uint8_t action,
                                  uint64_t pointer_id,
                                  uint32_t x,
                                  uint32_t y,
                                  uint16_t screen_width,
                                  uint16_t screen_height,
                                  float pressure,
                                  uint32_t action_button,
                                  uint32_t buttons) {
  // scrcpy protocol: type, action, pointer_id, x, y, screen_width, screen_height, pressure, action_button, buttons
  std::vector<uint8_t> msg;
  msg.push_back(2); // type = INJECT_TOUCH_EVENT
  msg.push_back(action);
  // pointer_id (u64, big-endian)
  for (int i = 7; i >= 0; i--)
    msg.push_back((pointer_id >> (i * 8)) & 0xFF);
  // x (u32, big-endian)
  msg.push_back((x >> 24) & 0xFF);
  msg.push_back((x >> 16) & 0xFF);
  msg.push_back((x >> 8) & 0xFF);
  msg.push_back(x & 0xFF);
  // y (u32, big-endian)
  msg.push_back((y >> 24) & 0xFF);
  msg.push_back((y >> 16) & 0xFF);
  msg.push_back((y >> 8) & 0xFF);
  msg.push_back(y & 0xFF);
  // screen_width (u16, big-endian)
  msg.push_back((screen_width >> 8) & 0xFF);
  msg.push_back(screen_width & 0xFF);
  // screen_height (u16, big-endian)
  msg.push_back((screen_height >> 8) & 0xFF);
  msg.push_back(screen_height & 0xFF);
  // pressure (u16, big-endian, 1.0f = 0xffff)
  uint16_t pressure_u16 =
      (pressure >= 1.0f) ? 0xffff : (pressure <= 0.0f ? 0 : static_cast<uint16_t>(pressure * 65535.0f));
  msg.push_back((pressure_u16 >> 8) & 0xFF);
  msg.push_back(pressure_u16 & 0xFF);
  // action_button (u32, big-endian)
  msg.push_back((action_button >> 24) & 0xFF);
  msg.push_back((action_button >> 16) & 0xFF);
  msg.push_back((action_button >> 8) & 0xFF);
  msg.push_back(action_button & 0xFF);
  // buttons (u32, big-endian)
  msg.push_back((buttons >> 24) & 0xFF);
  msg.push_back((buttons >> 16) & 0xFF);
  msg.push_back((buttons >> 8) & 0xFF);
  msg.push_back(buttons & 0xFF);
  if (!send_control_message(msg)) {
    std::cerr << "[ScrcpySink] Failed to send touch event" << std::endl;
  }
}
} // namespace vservo