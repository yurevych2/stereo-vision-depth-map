cmake_minimum_required(VERSION 3.25)
project(stereo_vision_depth_map)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)

add_executable(stereo_vision_depth_map
        src/smth.cpp
        src/smth.h
        main.cpp)
