# Configure version file using the project version
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/config.h.in"
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
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}/config>
)

set_target_properties(AppConfig PROPERTIES
    PUBLIC_HEADER "${CONFIG_HEADERS}"
)

install(TARGETS AppConfig
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION  lib
    ARCHIVE DESTINATION  lib
    RUNTIME DESTINATION  bin
    INCLUDES DESTINATION include/${PROJECT_NAME}/config
    PUBLIC_HEADER DESTINATION include/${PROJECT_NAME}/config
)
