#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
std::string get_tegra_pipeline(int width, int height, int fps) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(width) + ", height=(int)" +
           std::to_string(height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(fps) +
           "/1 ! nvvidconv flip-method=2 ! video/x-raw, width=960, height=616 ! nvvidconv ! nvegltransform ! nveglglessink -e";
}
// gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=1280, height=720, 
// framerate=120/1, format=NV12'! nvvidconv flip-method=2 ! 'video/x-raw,width=960,height=616'! nvvidconv ! nvegltransform ! nveglglessink -e

int main() {
    // Options
    int WIDTH = 1280;
    int HEIGHT = 720;
    int FPS = 120;

    // Define the gstream pipeline
    std::string pipeline = get_tegra_pipeline(WIDTH, HEIGHT, FPS);
    std::cout << "Using pipeline: \n\t" << pipeline << "\n";

    // Create OpenCV capture object, ensure it works.
    cv::VideoCapture cap("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)960, height=(int)616,format=(string)NV12, framerate=(fraction)120/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert !  appsink");

    // cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cout << "Connection failed";
        return -1;
    }

    // View video
    cv::Mat frame;
    while (1) {
        cap >> frame;  // Get a new frame from camera

        // Display frame
        imshow("Display window", frame);
        cv::waitKey(1); //needed to show frame
    }
}
