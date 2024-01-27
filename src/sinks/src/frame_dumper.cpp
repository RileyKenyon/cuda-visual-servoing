///-----------------------------------------------------------------------------
/// @file frame_dumper.cpp
///
/// @brief Implementation of debug frame dumper
///
/// @date 2026-02-17
///-----------------------------------------------------------------------------
#include "frame_dumper.hpp"
#include <iostream>
#include <stdexcept>

extern "C" {
#include <libswscale/swscale.h>
}

namespace vservo {

FrameDumper::FrameDumper(const std::string &output_file, uint32_t width, uint32_t height, int fps)
    : output_file_(output_file), width_(width), height_(height), fps_(fps) {
}

FrameDumper::~FrameDumper() {
    stop();
    close_output();
}

void FrameDumper::start() {
    if (running_) {
        return;
    }

    if (!open_output()) {
        throw std::runtime_error("Failed to open output file: " + output_file_);
    }

    running_ = true;
    stopped_ = false;
    dump_thread_ = std::thread(&FrameDumper::dump_thread, this);
    std::cout << "[FrameDumper] Started writing to " << output_file_ << std::endl;
}

void FrameDumper::stop() {
    if (!running_) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
    }
    cond_.notify_one();

    if (dump_thread_.joinable()) {
        dump_thread_.join();
    }

    // Write trailer
    if (format_ctx_ && header_written_) {
        av_write_trailer(format_ctx_);
    }

    running_ = false;
    std::cout << "[FrameDumper] Stopped" << std::endl;
}

bool FrameDumper::push_frame(AVFrame *frame) {
    if (!running_ || !frame) {
        return false;
    }

    // Clone the frame
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

bool FrameDumper::open_output() {
    // Allocate output format context
    const char *format_name = "mp4";
    avformat_alloc_output_context2(&format_ctx_, nullptr, format_name, output_file_.c_str());
    if (!format_ctx_) {
        std::cerr << "[FrameDumper] Failed to allocate output context" << std::endl;
        return false;
    }

    // Find encoder
    const AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        std::cerr << "[FrameDumper] H.264 codec not found" << std::endl;
        avformat_free_context(format_ctx_);
        format_ctx_ = nullptr;
        return false;
    }

    // Create output stream
    AVStream *stream = avformat_new_stream(format_ctx_, codec);
    if (!stream) {
        std::cerr << "[FrameDumper] Failed to create stream" << std::endl;
        avformat_free_context(format_ctx_);
        format_ctx_ = nullptr;
        return false;
    }

    // Allocate codec context
    encoder_ctx_ = avcodec_alloc_context3(codec);
    if (!encoder_ctx_) {
        avformat_free_context(format_ctx_);
        format_ctx_ = nullptr;
        return false;
    }

    // Configure encoder
    encoder_ctx_->width = width_;
    encoder_ctx_->height = height_;
    encoder_ctx_->pix_fmt = AV_PIX_FMT_YUV420P;
    encoder_ctx_->time_base = {1, fps_};
    encoder_ctx_->framerate = {fps_, 1};
    encoder_ctx_->gop_size = 10;

    // Apply format options
    if (format_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
        encoder_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    // Open encoder
    if (avcodec_open2(encoder_ctx_, codec, nullptr) < 0) {
        std::cerr << "[FrameDumper] Failed to open encoder" << std::endl;
        avcodec_free_context(&encoder_ctx_);
        avformat_free_context(format_ctx_);
        format_ctx_ = nullptr;
        encoder_ctx_ = nullptr;
        return false;
    }

    // Copy codec parameters to stream
    avcodec_parameters_from_context(stream->codecpar, encoder_ctx_);
    stream->time_base = encoder_ctx_->time_base;

    // Open output file
    if (!(format_ctx_->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&format_ctx_->pb, output_file_.c_str(), AVIO_FLAG_WRITE) < 0) {
            std::cerr << "[FrameDumper] Failed to open output file" << std::endl;
            avcodec_free_context(&encoder_ctx_);
            avformat_free_context(format_ctx_);
            format_ctx_ = nullptr;
            encoder_ctx_ = nullptr;
            return false;
        }
    }

    // Allocate packet
    packet_ = av_packet_alloc();
    if (!packet_) {
        if (format_ctx_->pb) avio_close(format_ctx_->pb);
        avcodec_free_context(&encoder_ctx_);
        avformat_free_context(format_ctx_);
        format_ctx_ = nullptr;
        encoder_ctx_ = nullptr;
        return false;
    }

    return true;
}

void FrameDumper::close_output() {
    if (packet_) {
        av_packet_free(&packet_);
        packet_ = nullptr;
    }

    if (encoder_ctx_) {
        avcodec_free_context(&encoder_ctx_);
        encoder_ctx_ = nullptr;
    }

    if (format_ctx_) {
        if (format_ctx_->pb) {
            avio_close(format_ctx_->pb);
            format_ctx_->pb = nullptr;
        }
        avformat_free_context(format_ctx_);
        format_ctx_ = nullptr;
    }
}

void FrameDumper::dump_thread() {
    int frame_count = 0;

    while (true) {
        AVFrame *frame = nullptr;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_.wait(lock, [this]() { return !frame_queue_.empty() || stopped_; });

            if (stopped_ && frame_queue_.empty()) {
                break;
            }

            if (!frame_queue_.empty()) {
                frame = frame_queue_.front();
                frame_queue_.pop();
            }
        }

        if (frame) {
            // Write header before first frame
            if (!header_written_) {
                if (avformat_write_header(format_ctx_, nullptr) < 0) {
                    std::cerr << "[FrameDumper] Failed to write header" << std::endl;
                } else {
                    header_written_ = true;
                }
            }

            if (header_written_) {
                // Set frame timestamp
                frame->pts = frame_count;

                // Encode frame
                if (avcodec_send_frame(encoder_ctx_, frame) >= 0) {
                    while (avcodec_receive_packet(encoder_ctx_, packet_) >= 0) {
                        packet_->stream_index = 0;
                        av_packet_rescale_ts(packet_, encoder_ctx_->time_base, 
                                           format_ctx_->streams[0]->time_base);
                        av_interleaved_write_frame(format_ctx_, packet_);
                        av_packet_unref(packet_);
                    }
                }
                frame_count++;
            }

            av_frame_free(&frame);
        }
    }

    std::cout << "[FrameDumper] Dumped " << frame_count << " frames to " << output_file_ << std::endl;
}

} // namespace vservo
