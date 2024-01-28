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
add_library(cvs::AppConfig ALIAS AppConfig)

# Set public headers
set(CONFIG_HEADERS
    include/config.h
    include/errors.h
)

target_include_directories(AppConfig INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

set_target_properties(AppConfig PROPERTIES
    PUBLIC_HEADER "${CONFIG_HEADERS}"
)

install(TARGETS AppConfig
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION  lib
    ARCHIVE DESTINATION  lib
    RUNTIME DESTINATION  bin
    INCLUDES DESTINATION include
)

# Add subdirectories
add_subdirectory(src/camera)
add_subdirectory(examples)


# Export
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(EXPORT ${PROJECT_NAME}Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE cvs::
        DESTINATION lib/cmake/${PROJECT_NAME}
)


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