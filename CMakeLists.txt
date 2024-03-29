cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
# Windows 系统下指定 pytorch 路径和 opencv 路径
# set(Torch_DIR /path/to/libtorch/share/cmake/Torch)
# set(OpenCV_DIR /path/to/opencv/build)

project(YoloLibTorch)

include_directories(./include)

add_subdirectory(./src)
add_subdirectory(./test)

install(TARGETS test Yolo
    RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin
    ARCHIVE DESTINATION ${CMAKE_SOURCE_DIR}/lib
)