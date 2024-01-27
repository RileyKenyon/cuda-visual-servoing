///-----------------------------------------------------------------------------
/// @file frame_dumper.hpp
///
/// @brief Debug utility to dump decoded frames to disk for inspection
///
/// @date 2026-02-17
///-----------------------------------------------------------------------------
#ifndef FRAME_DUMPER_HPP
#define FRAME_DUMPER_HPP

#include <cstdint>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
}

namespace vservo {

/// Simple frame dumper - writes decoded frames to MP4 file for inspection
class FrameDumper {
public:
    /// Create frame dumper for output file
    /// @param output_file Path to output MP4 file (e.g., "/tmp/frames.mp4")
    /// @param width Frame width
    /// @param height Frame height
    /// @param fps Frames per second
    explicit FrameDumper(const std::string &output_file, uint32_t width, uint32_t height, int fps = 30);

    /// Destructor
    ~FrameDumper();

    /// Start dumping frames
    void start();

    /// Stop dumping frames and finalize file
    void stop();

    /// Push a frame to be dumped
    bool push_frame(AVFrame *frame);

    /// Check if dumper is running
    bool is_running() const { return running_; }

private:
    bool open_output();
    void close_output();
    void dump_thread();

    std::string output_file_;
    uint32_t width_;
    uint32_t height_;
    int fps_;

    AVFormatContext *format_ctx_{nullptr};
    AVCodecContext *encoder_ctx_{nullptr};
    AVPacket *packet_{nullptr};

    std::queue<AVFrame*> frame_queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
    std::thread dump_thread_;

    bool running_{false};
    bool stopped_{false};
    bool header_written_{false};
};

} // namespace vservo

#endif // FRAME_DUMPER_HPP
