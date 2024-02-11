cmake_minimum_required(VERSION 3.16)
project(TestApp 
      VERSION 0.1
      DESCRIPTION "Example usage of library")

find_package(VServo REQUIRED)

# Target
add_executable(CameraApp
    main.cpp
)

# Add binary tree to search path for include files
target_include_directories(CameraApp 
    PUBLIC "${PROJECT_BINARY_DIR}/include"
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# Link example to the camera library
target_link_libraries(CameraApp PUBLIC vservo::AppConfig opencv_highgui)

# Request that CameraApp be built with -std=c++11
# As this is a public compile feature anything that links to CameraApp
# will also build with -std=c++11
target_compile_features(CameraApp PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the CameraApp
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties(CameraApp
    PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    ARCHIVE_OUTPUT_DIRECTORY lib
    LIBRARY_OUTPUT_DIRECTORY lib
    RUNTIME_OUTPUT_DIRECTORY bin
)
