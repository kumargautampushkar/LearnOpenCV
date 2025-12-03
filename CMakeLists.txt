cmake_minimum_required(VERSION 3.10)

# Project Name
project(FedoraOpenCVTest)

# OpenCV 4 requires at least C++11, but C++14 or 17 is recommended for modern features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------
# 1. Find the OpenCV Package
# On Fedora, this looks in /usr/lib64/cmake/opencv4/
# ------------------------------------------------------------------
find_package(OpenCV REQUIRED)

# Debugging: Print found paths to the terminal so you know it worked
message(STATUS "OpenCV found at: ${OpenCV_DIR}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Include dirs: ${OpenCV_INCLUDE_DIRS}")

# ------------------------------------------------------------------
# 2. Setup Includes and Linking
# ------------------------------------------------------------------

# This line specifically fixes the "fatal error: opencv2/opencv.hpp" 
# by including /usr/include/opencv4 automatically.
include_directories(${OpenCV_INCLUDE_DIRS})

# Create the executable (Replace 'main.cpp' with your file name)
add_executable(MyRobotApp main.cpp ObjectDetector.cpp)
add_executable(SiftObjDetector SiftObjDetector.cpp)

# Link the executable to the OpenCV libraries found
target_link_libraries(MyRobotApp ${OpenCV_LIBS})
target_link_libraries(SiftObjDetector ${OpenCV_LIBS})