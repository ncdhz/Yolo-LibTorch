find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

add_library(YoloV5 YoloV5.cpp)

target_link_libraries(YoloV5 ${OpenCV_LIBS} )
target_link_libraries(YoloV5 ${TORCH_LIBRARIES})