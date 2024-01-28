# Add Camera Library
add_library(camera
    src/camera.cpp
)

target_include_directories(camera PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}/camera>
)

# Add dependencies for opencv
target_link_libraries(camera PUBLIC opencv_core vservo::AppConfig)

# Request that cameralib be built with -std=c++11
# As this is a public compile feature anything that links to cameralib
# will also build with -std=c++11
target_compile_features(camera PUBLIC cxx_std_11)

# Add alias for building in source to look equivalent to using install
add_library(vservo::camera ALIAS camera)

set_target_properties(camera
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    PUBLIC_HEADER "include/camera.hpp"
)

install(TARGETS camera
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib/${PROJECT_NAME}
    ARCHIVE DESTINATION lib/${PROJECT_NAME}
    RUNTIME DESTINATION bin/${PROJECT_NAME}
    PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}/camera"
)