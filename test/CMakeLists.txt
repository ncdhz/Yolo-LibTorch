install(FILES ./coco.txt ./yolov8n.cpu.torchscript ./yolov8n.cuda.torchscript DESTINATION ${CMAKE_SOURCE_DIR}/bin)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

add_executable(test test.cpp)

target_link_libraries(test Yolo)
target_link_libraries(test ${OpenCV_LIBS} )
target_link_libraries(test ${TORCH_LIBRARIES})

