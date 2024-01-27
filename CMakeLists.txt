cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(VisualServoing LANGUAGES CXX VERSION 1.0)

include(GNUInstallDirs)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable CUDA if available
include(CheckLanguage)
check_language(CUDA)
if (DEFINED CMAKE_CUDA_COMPILER)
    # enable_language(CUDA)
else()
    message(STATUS "No CUDA support")
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)

# Configure version file using the project version
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/config.h"
)

add_library(AppConfig INTERFACE)
target_include_directories(AppConfig INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/include>
)

# Add subdirectories
add_subdirectory(src/camera)
add_subdirectory(examples)

# include(CTest)

# if(BUILD_TESTING)

#   add_executable(particle_test test.cu)

#   set_target_properties(particle_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#   target_link_libraries(particle_test PRIVATE CameraApp)

#   add_test(NAME camera_10k COMMAND particle_test 10000 )
#   add_test(NAME camera_256k COMMAND particle_test 256000 )

#   if(APPLE)
#     # We need to add the default path to the driver (libcuda.dylib) as an rpath,
#     # so that the static cuda runtime can find it at runtime.
#     set_property(TARGET particle_test PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
#   endif()
# endif()