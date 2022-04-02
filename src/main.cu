#include <stdlib.h>
#include "config.h"
#include "camera.hpp"


int main(int argc, char* argv[]) {
    camera::TegraCamera* cam = nullptr;
    if (argc < 2)
    {
        cam = new camera::TegraCamera();
    } 
    else if (4 == argc)
    {
        uint32_t u32_width = (uint32_t) std::atoi(argv[1]);
        uint32_t u32_height = (uint32_t) std::atoi(argv[2]);
        uint8_t u8_fps = (uint8_t) std::atoi(argv[3]);
        std::cout << "Width: " << std::to_string(u32_width) << std::endl;
        std::cout << "Height: " << std::to_string(u32_height) << std::endl;
        std::cout << "FPS: " << std::to_string(u8_fps) << std::endl;
        cam = new camera::TegraCamera(u32_width, u32_height, u8_fps);
    }
    else
    {
        // report out the version
        std::cout << argv[0] << " Version " << VisualServoing_VERSION_MAJOR << "."
                  << VisualServoing_VERSION_MINOR << std::endl;
        std::cout << "Usage: " << argv[0] << " number" << std::endl;
        return -1;
    }

    // Video Capture
    cam->open_gst();
    cv::Mat frameIn;

    for(;;)
    {
        // error handling
        if (!cam->read(frameIn)) {
            std::cout<<"Capture read error"<<std::endl;
            break;
        }

        // show the image
        else {
            cv::imshow("MyCameraPreview",frameIn);
            if(cv::waitKey(1) >= 0){
                printf("Exiting.");
                break;
            }
        } 
    }
    cam->release();
    delete cam;
    return 0;
}