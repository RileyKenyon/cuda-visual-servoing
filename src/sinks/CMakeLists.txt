add_library(sinks 
    src/scrcpy_sink.cpp
    src/v4l2_sink.cpp
)
add_library(vservo::sinks ALIAS sinks)


include(GNUInstallDirs)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

target_include_directories(sinks PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}/sinks>
)

# Compiler flags for C compatibility
target_compile_options(sinks PRIVATE -fpermissive)
target_compile_features(sinks PRIVATE cxx_std_17)

# Find and link FFmpeg packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
    libavcodec
    libavformat
    libavdevice
    libavutil
    libswscale
)

target_link_libraries(sinks 
    PRIVATE PkgConfig::LIBAV
)

# install(TARGETS sinks
#     EXPORT ${PROJECT_NAME}Targets
#     LIBRARY DESTINATION lib/${PROJECT_NAME}
#     ARCHIVE DESTINATION lib/${PROJECT_NAME}
#     RUNTIME DESTINATION bin/${PROJECT_NAME}
#     PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}/sinks"
# )