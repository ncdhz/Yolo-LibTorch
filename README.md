### 使用方法

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
        cv::Mat img = yolo.drawRectangle("图片", r[0]);
        ```
    + 判断是否存在类型
        ```
        bool is = yolo.existencePrediction(r);
        ```
5. 其余方法使用（见`.h`文件注解）