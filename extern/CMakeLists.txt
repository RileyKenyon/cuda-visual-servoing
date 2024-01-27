# cmake_minimum_required(VERSION 3.16)

# project(scrcpy
#   LANGUAGES C CXX
# )

set(SCRCPY_SRC
  scrcpy/app/src/adb/adb.c
  scrcpy/app/src/adb/adb_device.c
  scrcpy/app/src/adb/adb_parser.c
  scrcpy/app/src/adb/adb_tunnel.c
  scrcpy/app/src/audio_player.c
  scrcpy/app/src/cli.c
  scrcpy/app/src/clock.c
  scrcpy/app/src/compat.c
  scrcpy/app/src/control_msg.c
  scrcpy/app/src/controller.c
  scrcpy/app/src/decoder.c
  scrcpy/app/src/delay_buffer.c
  scrcpy/app/src/demuxer.c
  scrcpy/app/src/device_msg.c
  scrcpy/app/src/display.c
  scrcpy/app/src/icon.c
  scrcpy/app/src/file_pusher.c
  scrcpy/app/src/fps_counter.c
  scrcpy/app/src/frame_buffer.c
  scrcpy/app/src/input_manager.c
  scrcpy/app/src/keyboard_inject.c
  scrcpy/app/src/mouse_inject.c
  scrcpy/app/src/opengl.c
  scrcpy/app/src/options.c
  scrcpy/app/src/packet_merger.c
  scrcpy/app/src/receiver.c
  scrcpy/app/src/recorder.c
  scrcpy/app/src/screen.c
  scrcpy/app/src/version.c
  scrcpy/app/src/trait/frame_source.c
  scrcpy/app/src/trait/packet_source.c
  scrcpy/app/src/util/acksync.c
  scrcpy/app/src/util/average.c
  scrcpy/app/src/util/bytebuf.c
  scrcpy/app/src/util/file.c
  scrcpy/app/src/util/intmap.c
  scrcpy/app/src/util/intr.c
  scrcpy/app/src/util/log.c
  scrcpy/app/src/util/memory.c
  scrcpy/app/src/util/net.c
  scrcpy/app/src/util/net_intr.c
  scrcpy/app/src/util/process_intr.c
  scrcpy/app/src/util/rand.c
  scrcpy/app/src/util/strbuf.c
  scrcpy/app/src/util/str.c
  scrcpy/app/src/util/term.c
  scrcpy/app/src/util/thread.c
  scrcpy/app/src/util/tick.c
  scrcpy/app/src/util/timeout.c
  scrcpy/app/src/usb/aoa_hid.c
  scrcpy/app/src/usb/hid_keyboard.c
  scrcpy/app/src/usb/hid_mouse.c
  scrcpy/app/src/usb/scrcpy_otg.c
  scrcpy/app/src/usb/screen_otg.c
  scrcpy/app/src/usb/usb.c
  scrcpy/app/src/util/process.c
  scrcpy/app/src/sys/unix/file.c
  scrcpy/app/src/sys/unix/process.c
  scrcpy/app/src/v4l2_sink.c
  scrcpy/app/src/server.c
  scrcpy/app/src/scrcpy.c
)

add_library(scrcpy SHARED ${SCRCPY_SRC})

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
    libavdevice
    libavfilter
    libavformat
    libavcodec
    libswresample
    libswscale
    libavutil
)

target_compile_definitions(scrcpy PUBLIC -Dprebuilt_server=scrcpy-server)

# set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})

# include(cmake/FindLIBAV.cmake)
# list(APPEND ${CMAKE_MODULE_PATH} cmake)
find_package(SDL2 REQUIRED)
target_link_libraries(scrcpy PUBLIC ${SDL2_LIBRARIES} ${LIBAV_LIBRARIES} usb-1.0)
target_include_directories(scrcpy PUBLIC
  scrcpy/app/src
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${LIBAV_INCLUDE_DIRS}
  ${SDL2_INCLUDE_DIRS}
)
add_executable(scrcpy_app scrcpy/app/src/main.c)
target_link_libraries(scrcpy_app PUBLIC scrcpy)