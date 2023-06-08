find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

add_library(Yolo Yolo.cpp)

target_link_libraries(Yolo ${OpenCV_LIBS} )
target_link_libraries(Yolo ${TORCH_LIBRARIES})