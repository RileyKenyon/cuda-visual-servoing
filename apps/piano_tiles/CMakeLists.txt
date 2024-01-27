# Add Applications
add_executable(PianoTiles main.cpp)
target_compile_features(PianoTiles PRIVATE cxx_std_17)

# Add dependencies
target_link_libraries(PianoTiles PUBLIC 
    opencv_core 
    vservo::image_processing
    vservo::sinks
)

install(
    TARGETS PianoTiles
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib/${PROJECT_NAME}
    ARCHIVE DESTINATION lib/${PROJECT_NAME}
    RUNTIME DESTINATION bin/${PROJECT_NAME}
)