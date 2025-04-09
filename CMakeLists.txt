cmake_minimum_required(VERSION 3.25)
project(stereo_vision_depth_map)

set(CMAKE_CXX_STANDARD 17)

# Include your directories
include_directories(src lib img)

# Add your executable
add_executable(stereo_vision_depth_map
        src/smth.cpp
        src/smth.h
        lib/stb_image.h
        main.cpp)

# === OpenMP Support ===
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(stereo_vision_depth_map PUBLIC OpenMP::OpenMP_CXX)
endif()
