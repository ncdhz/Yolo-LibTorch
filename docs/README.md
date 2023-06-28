
## 文档

<a href="https://ncdhz.github.io/Yolo-LibTorch">Yolo-LibTorch</a>旨在使用<a href="https://pytorch.org/">LibTorch</a>轻松调用`Yolo`的各种版本<a href="https://github.com/ultralytics/yolov5">YoloV5</a>、<a href="https://github.com/meituan/YOLOv6">YoloV6</a>、<a href="https://github.com/WongKinYiu/yolov7">YoloV7</a>和<a href="https://github.com/ultralytics/ultralytics">YoloV8</a>等（以下结果通过`yolov8n`模型预测）。**此项目需要`torchscript`的模型格式**。
<br/>
<img src="./images/test.jpg" class="img"/>

## 快速开始

### Windows 中使用（cmake）

1. 下载所需环境
    + [cmake](https://cmake.org/)
    + [libtorch](https://pytorch.org/)
    + [opencv](https://opencv.org/releases/)
    + [visual studio](https://visualstudio.microsoft.com/)需要它提供的c++环境

2. 配置环境变量

    ```shell
    # 添加libtorch的lib目录到path
    \path\to\libtorch\lib
    # 添加opencv的bin目录到环境变量
    # vc14 vc15 vc16 都可以
    \path\to\opencv\build\x64\vc16\bin
    ```

3. 修改根目录下 `CMakeLists.txt` 文件

    ```shell
    # Torch_DIR 路径为 libtorch 下面的 share/cmake/Torch/
    set(Torch_DIR /path/to/libtorch/share/cmake/Torch/)
    # OpenCV_DIR 路径为 opencv/build/
    set(OpenCV_DIR /path/to/opencv/build/)
    ```

4. 编译

    ```shell
    mkdir build
    cd build
    cmake ..

    # 根据你 LibTorch 版本选择执行下面语句
    # Debug
    cmake --build . --config Debug
    cmake --install . --config Debug

    # Release
    cmake --build . --config Release
    cmake --install . --config Release
    ```

5. 测试

    ```shell
    cd bin
    # 默认会打开电脑摄像头捕捉画面
    # 如果电脑没有摄像头可以通过参数引入视频或图片
    test.exe
    ```
