#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <string>

#define NUM 10000
#define Frames 120
//GPU KERNEL
//----------------------------------------------------------------
__global__ void gpu_grayscale(unsigned char *matA,unsigned char *grayData, int width, int height){
  //Distance between array elements (i,j)[0] to (i,j)[1] is 1 not width*height
  //thread ID
  int tid;
  tid = blockIdx.x*blockDim.x + threadIdx.x;
  
  //stride lengths
  int stride;
  stride = blockDim.x*gridDim.x;

  // grayscale calculation with strides
  while (tid < width*height){
    grayData[tid] = matA[3*tid]*0.07 + matA[3*tid+1]*0.72 + matA[3*tid+2]*0.21;
    if (grayData[tid] < 128 ){ //saturate to either 255 or 0 - for pixel testing
      grayData[tid] = 255;
    } else {
      grayData[tid] = 0;
    }
    tid = tid + stride;
  }
  
}

// MAIN FUNCTION
//-----------------------------------------------------------------
//using namespace cv;
int main()
{
  // Initialize timer settings
  float calcTimer = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  //float GPUtimer, CPUtimer;

// ORIGINAL IMAGE
//-------------------------------------------------------------------
  // Get initial image and print
  cv::Mat img;
  //img = cv::imread("pineapple.jpeg");
  cv::VideoCapture cap("example.mp4"); // replace with 1 if using webcam
  //cv::VideoCapture cap(1);
  if (!cap.isOpened()){
    printf("Error getting Stream \n");
  }
  cap >> img;
  //cv::cvtColor(img,img,cv::COLOR_RGB2GRAY);
  cv::imwrite("Screen_Gray.png",img);
  //cv::imshow("original",img);
  //cv::waitKey();

// CALCULATION SETTINGS
//-------------------------------------------------------------------  
  // Size of windows  
  int width = img.cols;
  int height = img.rows;
  printf("Resolution: %d x %d \n",width, height);
  // Configure blocks and threads for GPU
  unsigned int numThreads, numBlocks;
  numThreads = 1024; // good number for multiple of 32
  numBlocks = (width*height + numThreads - 1)/numThreads; // make so at maximum only one additional block

  // Allocate device and host
  unsigned char *matA, *grayData;
  //cudaMalloc((void **) &matA, sizeof(unsigned char)*width*height*3);
  //cudaMalloc((void **) &grayData, sizeof(unsigned char)*width*height);
  cudaMallocManaged(&matA,sizeof(unsigned char)*width*height*3);
  cudaMallocManaged(&grayData,sizeof(unsigned char)*width*height);
  unsigned char *cpuData = (unsigned char *) malloc(width*height*sizeof(unsigned char));
  //Edge detection
  cv::Mat kernel = (cv::Mat_<int>(3,2) <<
      1,1,
      0,0,
      0,0); 
  cv::Mat outputGPU;
  
// GPU CALCULATION
//-----------------------------------------------------------------
  //cudaEventRecord(start);
  for (;;){
    cudaEventRecord(start);
    char c;
    for (int j = 0; j< Frames; j++){
      cap >> img;
      cudaMemcpy(matA, img.data, width*height*3* sizeof(unsigned char), cudaMemcpyHostToDevice); // NEED THIS LINE FOR COPYING ARRAY
      gpu_grayscale<<<numBlocks,numThreads>>>(matA,grayData,width,height);
      cudaDeviceSynchronize(); // sync threads and transfer memory
    // Show grayscale image
      //cv::Mat outputGPU(cv::Size(width,height),CV_8UC1,grayData);
      cv::Mat build(cv::Size(width,height),CV_8UC1,grayData); // NEED TO CHANGE TO GPU DATA IF USING CUDAMALLOC AND NOT CUDAMALLOCMANAGED
      // implement edge detect
      cv::morphologyEx(build,outputGPU,cv::MORPH_HITMISS,kernel);  
      //cv::rotate(outputGPU,outputGPU,cv::ROTATE_90_COUNTERCLOCKWISE); // for rotating recorded video    
      cv::imshow("GPU",outputGPU);
      c = cv::waitKey(1);
      if (c ==' ')
        break;
    }
    if (c ==' ')
      break;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&calcTimer,start, stop);
    printf("FPS GPU: %f \n",Frames/calcTimer*1000);
    calcTimer = 0;
  }
/**
// SPLITTING CALCULATION
//-------------------------------------------------------------------
  // Format Channels (shows only gray when looking at one array)
  cudaEventRecord(start);
  cv::Mat channels[3];  
  for (int i = 0; i < NUM; i++){
  cv::split(img,channels);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&calcTimer,start, stop);
  printf("Time (ms) for Splitting channel: %f\n",calcTimer/NUM);
  calcTimer = 0;
  // show first channel
  cv::imshow("one-channel",channels[0]);
  cv::waitKey();

**/
// CPU CALCULATION
//--------------------------------------------------------------------
  for (;;){
    cudaEventRecord(start);
    char c;
    for (int j = 0; j<Frames; j++){
      cap >> img;
      // cpu grayscale
      for (int i = 0; i < width*height; i++){
        cpuData[i] = img.data[3*i]*0.07 + img.data[3*i+1]*0.72 + img.data[3*i+2]*0.21;
        if (cpuData[i] > 128){ //saturate to either 255 or 0 - for pixel testing
          cpuData[i] = 255;
        } else {
          cpuData[i] = 0;
        }
      }
      // show grayscale
      cv::Mat reconstruction(cv::Size(img.cols,img.rows),CV_8UC1,cpuData);
      cv::Mat outputCPU;
      cv::morphologyEx(reconstruction,outputCPU,cv::MORPH_HITMISS,kernel); 
      //cv::rotate(outputCPU,outputCPU,cv::ROTATE_90_COUNTERCLOCKWISE); // for rotating recorded video
      cv::imshow("cpu_gray",outputCPU);
      c = cv::waitKey(1);
      if (c ==' ')
        break;
    }
    if (c ==' ')
      break;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&calcTimer,start, stop);
    printf("FPS CPU: %f\n",Frames/calcTimer*1000);
    calcTimer = 0;
  }
// CLOSEOUT
//--------------------------------------------------------------------------
  cudaFree(matA);
  cudaFree(grayData);
  free(cpuData);
}

/**
  // output dimensions of image
  std::cout <<"Size:	"<< img.rows<< "x"<< img.cols<< std::endl;

  // Blank Mat
  cv::Mat blank;
  blank = cv::Mat::zeros(cv::Size(img.cols,img.rows), CV_8UC3);

  // specify arguments of mixChannel
  int from_to[] = {0,0}; // changing first channel of blank to first channel of img

  // Time operation for image splicing
  t = (double)cv::getTickCount(); // time initialize
  cv::mixChannels(&img,1,&blank,1,from_to,1); //Left off HERE
  t = ((double)cv::getTickCount() - t)/cv::getTickFrequency(); // calculate in seconds
  std::cout << "Times passed in seconds: " << t << std::endl;
  cv::imshow("blue",blank);
  cv::waitKey();
  

  //std::cout << blue << std::endl;

**/
/**
  //cv::Mat cpuGray;
  //cpuGray = cv::Mat::zeros(cv::Size(img.cols,img.rows),CV_8UC1);
  //cv::Mat_<cv::Vec3b> &temp = reinterpret_cast<cv::Mat_<cv::Vec3b>&>(img); // Appropriately type pixels - allows for temp(j,i)[k] for k =0,1,2
  // Slower CPU calculation with convoluted access
  for (int i = 0; i < img.rows; i++){
    for (int j = 0; j < img.cols; j++){
    cpuGray.at<uchar>(j,i) = temp(j,i)[0]*0.07 + temp(j,i)[1]*0.72 + temp(j,i)[2]*0.21;
    }
  }
**/
/**
  // Time operation for splitting
  double t = (double)cv::getTickCount(); // time initialize
  cv::split(img,channels);
  t = ((double)cv::getTickCount() - t)/cv::getTickFrequency(); // calculate in seconds
  std::cout << "Time (seconds) for splitting channel: " << t << std::endl;
**/
