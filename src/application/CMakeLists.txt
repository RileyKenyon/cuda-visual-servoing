# Add Applications
add_executable(MiddleAlgorithm middle.cu)
add_executable(PixelAlgorithm pixel.cu)
add_executable(ReductionAlgorithm reduction.cu)
add_executable(TriggerApplication trigger.cu)
add_executable(VelocityApplication velocity.cu)

# Add dependencies for opencv
target_link_libraries(MiddleAlgorithm PUBLIC opencv_core vservo::image_processing)
target_compile_features(MiddleAlgorithm PRIVATE cxx_std_14)
target_link_libraries(PixelAlgorithm PUBLIC opencv_core vservo::image_processing)
target_link_libraries(ReductionAlgorithm PUBLIC opencv_core vservo::image_processing)
target_link_libraries(TriggerApplication PUBLIC opencv_core vservo::image_processing)
target_link_libraries(VelocityApplication PUBLIC opencv_core vservo::image_processing)

install(
    TARGETS 
      MiddleAlgorithm
      PixelAlgorithm
      ReductionAlgorithm
      TriggerApplication
      VelocityApplication
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib/${PROJECT_NAME}
    ARCHIVE DESTINATION lib/${PROJECT_NAME}
    RUNTIME DESTINATION bin/${PROJECT_NAME}
)