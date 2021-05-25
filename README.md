### Windows 10 中编译此项目
1. 下载所需要的编译工具
    
    ```
    cmake
    opencv
    libtorch
    ```

2. 打开此项目下面的 `CMakeLists.txt` 按下面的方式添加路径
    ```
    // Torch_DIR 路径为 libtorch 下面的 share/cmake/Torch/
    set(Torch_DIR /path/to/libtorch/share/cmake/Torch/)
    // OpenCV_DIR 路径为 opencv/build/
    set(OpenCV_DIR /path/to/opencv/build/)
    ```
3. 执行下面方法进行生成项目
    
    ```
    mkdir build
    cd build
    cmake ..
    // 根据你 LibTorch 版本选择执行下面语句
    // Debug
    cmake --build . --config Debug
    cmake --install . --config Debug
    // Release
    cmake --build . --config Release
    cmake --install . --config Release
    ```
4. 在 `Path` 中添加下面的环境变量
    
    ```
    \path\to\libtorch\lib
    // vc15 vc14 都可以
    \path\to\opencv\build\x64\vc15\bin
    ```
5. 测试
    ```
    cd bin
    test.exe
    ```
### Linux 等系统中编译此项目
由于没有测试不便多说应该是更加简单，安装好 opencv 和 libtorch 之后直接执行下面代码即可

```
mkdir build
cd build
cmake ..
make
make install
```
### 使用方法 1

直接在工程中引入刚才生成工程下 lib 目录（库目录）和 include 目录（头文件目录），具体 `API` 见头文件或者使用方法 2 中的一些阐述

### 使用方法 2

1. 添加 `YoloV5.cpp` 和 `YoloV5.h` 到你的项目
2. 构造 `YoloV5` 对象
    ```
    // pt 文件路径
    YoloV5 yolo("ptFile");
    ```
3. 预测
    + 直接图片预测
        ```
        cv::Mat img = cv::imread("图片路径");
        std::vector<torch::Tensor> r = yolo.prediction(img);
        ```
    + 图片路径预测
        ```
        std::vector<torch::Tensor> r = yolo.prediction("图片路径");
        ```
4. 对结果处理
    + 画框
        ```
        // img 为预测前的图片 
        cv::Mat img = yolo.drawRectangle(img, r[0]);
        ```
    + 判断是否存在类型
        ```
        bool is = yolo.existencePrediction(r);
        ```
5. 其余方法使用（见`.h`文件注解）