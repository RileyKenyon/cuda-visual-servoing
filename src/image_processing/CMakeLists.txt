# Add library for image processing
add_library(image_processing src/imgProc.cu)
add_library(cvs::image_processing ALIAS image_processing)

# Mandate any target linking needs to be compiled with -std=c++11
target_compile_features(image_processing PUBLIC cxx_std_11)

target_include_directories(image_processing PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}/image_processing>
)

target_link_libraries(image_processing PUBLIC 
    opencv_core 
    opencv_highgui
)

set (IMAGE_PROCESSING_HEADERS
    include/imgProc.hpp
)

set_target_properties(image_processing
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    PUBLIC_HEADER "${IMAGE_PROCESSING_HEADERS}"
)

install(TARGETS image_processing
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}/image_processing"
)