extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
}

#include <iostream>
#include <vector>

void analyze_luma_pixels(const uint8_t* luma, int linesize, int width, int height) {
    // Zero-copy direct pixel access
    // linesize accounts for memory alignment padding (stride >= width)
    uint64_t total_brightness = 0;

    for (int y = 0; y < height; ++y) {
        const uint8_t* row = luma + (y * linesize);
        for (int x = 0; x < width; ++x) {
            total_brightness += row[x];
        }
    }
    double avg_intensity = static_cast<double>(total_brightness) / (width * height);
    // Feed into your classifier/feature buffer here...
}

int main(int argc, char* argv[]) {
    if (argc < 2) return 1;
    const char* filename = argv[1];

    AVFormatContext* format_ctx = avformat_alloc_context();
    if (avformat_open_input(&format_ctx, filename, nullptr, nullptr) != 0) return -1;
    if (avformat_find_stream_info(format_ctx, nullptr) < 0) return -1;

    // 1. Locate the video stream
    int video_stream_idx = -1;
    const AVCodec* codec = nullptr;
    for (unsigned int i = 0; i < format_ctx->nb_streams; ++i) {
        if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx = i;
            codec = avcodec_find_decoder(format_ctx->streams[i]->codecpar->codec_id);
            break;
        }
    }
    if (video_stream_idx == -1 || !codec) return -1;

    // 2. Configure Decoder & Multi-threading
    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_ctx, format_ctx->streams[video_stream_idx]->codecpar);
    
    // Enable multi-threaded slice/frame decoding for max speed
    codec_ctx->thread_count = 0; // 0 lets FFmpeg auto-detect CPU cores

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) return -1;

    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();

    // 3. Low-level Demux & Decode Loop
    while (av_read_frame(format_ctx, packet) >= 0) {
        if (packet->stream_index == video_stream_idx) {
            if (avcodec_send_packet(codec_ctx, packet) == 0) {
                while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                    // Direct pointer to raw Y-channel plane
                    analyze_luma_pixels(frame->data[0], frame->linesize[0], frame->width, frame->height);
                }
            }
        }
        av_packet_unref(packet);
    }

    // Flush decoder
    avcodec_send_packet(codec_ctx, nullptr);
    while (avcodec_receive_frame(codec_ctx, frame) == 0) {
        analyze_luma_pixels(frame->data[0], frame->linesize[0], frame->width, frame->height);
    }

    // Cleanup
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&format_ctx);
    return 0;
}