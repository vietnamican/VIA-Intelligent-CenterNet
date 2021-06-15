cmake_minimum_required(VERSION 3.10)

SET(OpenCV_DIR "~/Workspace/opencv/build")
SET(ncnn_DIR "~/Workspace/ncnn/build")
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_STANDARD 11)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Debug)" FORCE)
endif()

project(Ultra-Light-Fast-Generic-Face-Detector-1MB)

add_subdirectory(../../ncnn ncnn_build)
include_directories(../../ncnn/src)
# add_library(centerface SHARED main)


find_package(OpenCV REQUIRED)

add_executable(main main.cpp CenterFace.cpp)
target_link_libraries(main ncnn ${OpenCV_LIBS})
