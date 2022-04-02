#include "config.h"
#include "camera.hpp"


int main(int argc, char* argv[]) {
    // report out the version
    if (argc < 2)
    {
        std::cout << argv[0] << " Version " << VisualServoing_VERSION_MAJOR << "."
                  << VisualServoing_VERSION_MINOR << std::endl;
        std::cout << "Usage: " << argv[0] << " number" << std::endl;
        return 1;
    }

    // Video Capture
    auto cam = camera::TegraCamera(960, 616, 120);
    cv::Mat frameIn;

    for(;;)
    {
        // error handling
        if (!cam.read(frameIn)) {
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
    cam.release();
    return 0;
}