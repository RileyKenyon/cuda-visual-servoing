cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(ImageProcessing LANGUAGES CXX CUDA VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


configure_file(
    "${PROJECT_SOURCE_DIR}/../../include/config.h.in"
    "${PROJECT_BINARY_DIR}/../../include/config.h"
    )

# Add dependencies for opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# List source
set(imgProc_SRC
    src/imgProc.cu
)

# List headers
set (imgProc_HEADERS
    include/imgProc.hpp
    ../../include/config.h
    ../../include/errors.h
)

add_library(imgProc_lib
    STATIC
    "${imgProc_SRC}"
    "${imgProc_HEADERS}"
)

target_link_libraries(imgProc_lib 
    "${OpenCV_LIBS}" 
    )

# Add binary tree to search path for include files
target_include_directories(imgProc_lib 
    PUBLIC "${PROJECT_BINARY_DIR}/include"
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )

# Request that imgProc_lib be built with -std=c++11
# As this is a public compile feature anything that links to cameralib
# will also build with -std=c++11
target_compile_features(imgProc_lib PUBLIC cxx_std_11)


# We need to explicitly state that we need all CUDA files in the cudacv
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties(imgProc_lib
    PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )