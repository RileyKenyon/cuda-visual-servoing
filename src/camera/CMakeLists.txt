cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(Camera LANGUAGES CXX VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable CUDA if able
include(CheckLanguage)
check_language(CUDA)
if (NOT DEFINED CMAKE_CUDA_COMPILER)
else()
    # enable_language(CUDA)
endif()

configure_file(
    "${PROJECT_SOURCE_DIR}/../../include/config.h.in"
    "${PROJECT_BINARY_DIR}/../../include/config.h"
    )
# Setup with conan
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# List source
set(camera_SRC
    src/camera.cpp
)

# List headers
set (camera_HEADERS
    include/camera.hpp
    ../../include/config.h
    ../../include/errors.h
)

add_library(camera_lib
    STATIC
    "${camera_SRC}"
    "${camera_HEADERS}"
)

# Add dependencies for opencv from conan
target_link_libraries(camera_lib
    ${CONAN_LIBS}
    )

# Add binary tree to search path for include files
target_include_directories(camera_lib 
    PUBLIC "${PROJECT_BINARY_DIR}/include"
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )

# Request that cameralib be built with -std=c++11
# As this is a public compile feature anything that links to cameralib
# will also build with -std=c++11
target_compile_features(camera_lib PUBLIC cxx_std_11)


# We need to explicitly state that we need all CUDA files in the cudacv
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties(camera_lib
    PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )