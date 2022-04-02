#include "camera.h"

std::string get_tegra_gst_pipeline(const uint32_t width, const uint32_t height, const uint8_t fps)
{
    return ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)960, height=(int)616,format=(string)NV12, \
            framerate=(fraction)" + std::to_string(fps) + "/1 ! nvvidconv ! video/x-raw, format=(string)BGRx \
            ! videoconvert !  appsink");
}