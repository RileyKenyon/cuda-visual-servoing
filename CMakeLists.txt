cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(VisualServoing LANGUAGES CXX CUDA VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

configure_file(
    "${PROJECT_SOURCE_DIR}/include/config.h.in"
    "${PROJECT_BINARY_DIR}/include/config.h"
    )

# Add dependencies for opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add subdirectories
add_subdirectory(
    src/camera
    src/imageProcessing
    )

# Target
add_executable(cudacv
    src/main.cu
    )

# Add binary tree to search path for include files
target_include_directories(cudacv 
    PUBLIC "${PROJECT_BINARY_DIR}/include"
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)


target_link_libraries(cudacv
    camera_lib
)

# Request that cudacv be built with -std=c++11
# As this is a public compile feature anything that links to cudacv
# will also build with -std=c++11
target_compile_features(cudacv PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the cudacv
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties(cudacv
    PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

# include(CTest)

# if(BUILD_TESTING)

#   add_executable(particle_test test.cu)

#   set_target_properties(particle_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#   target_link_libraries(particle_test PRIVATE cudacv)

#   add_test(NAME camera_10k COMMAND particle_test 10000 )
#   add_test(NAME camera_256k COMMAND particle_test 256000 )

#   if(APPLE)
#     # We need to add the default path to the driver (libcuda.dylib) as an rpath,
#     # so that the static cuda runtime can find it at runtime.
#     set_property(TARGET particle_test PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
#   endif()
# endif()