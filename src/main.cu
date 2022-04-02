#include "config.h"
#include "camera.h"


int main(int argc, char* argv[]) {
    if (argc < 2)
    {
        // report out the version
        std::cout << argv[0] << " Version " << VisualServoing_VERSION_MAJOR << "."
                  << VisualServoing_VERSION_MINOR << std::endl;
        std::cout << "Usage: " << argv[0] << " number" << std::endl;
        return 1;
    }

    // Video Capture
    std::string pipeline = get_tegra_gst_pipeline(CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS);
    cv::VideoCapture cap(pipeline);

    // for(;;)
    // {
    //     // error handling
    //     if (!cap.read(frame_in)) {
    //         std::cout<<"Capture read error"<<std::endl;
    //         break;
    //     }

    //     // show the image
    //     else {
    //         cv::imshow("MyCameraPreview",frame_in);
    //         if(cv::waitKey(1) >= 0){
    //             printf("Exiting.");
    //             break;
    //         }
    //     }
    // }

    cap.release();

    return 0;
}