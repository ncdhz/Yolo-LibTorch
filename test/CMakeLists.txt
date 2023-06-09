install(FILES ./resource/coco.txt ./resource/yolov8n.cpu.torchscript ./resource/yolov8n.cuda.torchscript DESTINATION ${CMAKE_SOURCE_DIR}/bin)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

add_executable(test test.cpp Parameter.cpp Parameter.h)

target_link_libraries(test ${OpenCV_LIBS} )
target_link_libraries(test ${TORCH_LIBRARIES})
target_link_libraries(test Yolo)
