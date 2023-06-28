## Yolo

### 初始化

```cpp
Yolo(std::string ptFile, std::string version="v8", std::string device="cpu", bool isHalf=false, 
    int height=640, int width=640, float confThres=0.25, float iouThres=0.45)
```

* `ptFile` 模型文件路径
* `version` 版本`["v5", "v6", "v7", "v8"]`中选一
* `isHalf` 是否使用半精度
* `device` 推理使用的设备默认为`cpu`可选`["cpu", "cuda:0", "cuda:1", ...]`
* `height` 训练时图片的高
* `width` 训练时图片的宽
* `confThres` 非极大值抑制中的 `scoreThresh`
* `iouThres` 非极大值抑制中的 `iouThresh`

> 初始化能获取一个`Yolo`对象。

### 预测

1. 方法一

    ```cpp
    std::vector<torch::Tensor> prediction(torch::Tensor data)
    ```

   * `data` 需要预测的`tensor`数据

    > 数据格式 `(batch, rgb, height, width)` `（批数量，颜色，高，宽）`。

2. 方法二

    ```cpp
    std::vector<torch::Tensor> prediction(std::string filePath)
    ```

   * `filePath` 文件路径

3. 方法三

    ```cpp
    std::vector<torch::Tensor> prediction(cv::Mat img)
    ```

   * `img` `opencv`的图片格式

4. 方法四

    ```cpp
    std::vector<torch::Tensor> prediction(std::vector<cv::Mat> imgs)
    ```

   * `imgs` 需要预测图片的集合

#### 返回值

* `std::vector<torch::Tensor>` 一个`tensor`集合

> 集合中每一个元素表示一张图片。其中每个`tensor`的维度是`6*n`，`6`表示`（左上点x坐标，左上点y坐标，右下点x坐标，右下点y坐标，置信度，标签）`，`n`表示预测出了多少个物体。

### 图片尺寸

1. 方法一

    ```cpp
    static ImageResizeData resize(cv::Mat img, int height, int width)
    ```

    * `img` 图片
    * `height` 高度
    * `width` 宽度

2. 方法二

    ```cpp
    ImageResizeData resize(cv::Mat img)
    ```

    * `img` 图片

    > `height`使用`Yolo`初始化的高度，`width`使用`yolo`初始化宽度。

3. 方法三

    ```cpp
    static std::vector<ImageResizeData> resize(std::vector <cv::Mat> imgs, int height, int width)
    ```

    * `imgs` 图片集合
    * `height` 高度
    * `width` 宽度

4. 方法四

    ```cpp
    std::vector<ImageResizeData> resize(std::vector <cv::Mat> imgs)
    ```

    * `imgs` 图片集合

    > `height`使用`Yolo`初始化的高度，`width`使用`yolo`初始化宽度。

#### 返回值

* [`ImageResizeData`](#imageresizedata) 图片尺寸数据类
* [`std::vector<ImageResizeData>`](#imageresizedata) 图片尺寸数据类集合

### 绘制矩形

1. 方法一

    ```cpp
    std::vector<cv::Mat> drawRectangle(std::vector<cv::Mat> imgs,
        std::vector<torch::Tensor> rectangles, std::map<int, std::string> labels, int thickness = 2)
    ```

    * `imgs` 需要绘制的图片集合
    * `rectangles` 预测的返回值
    * `labels` 标签字典
        * `key` 标签编号从0开始
        * `value` 编号对应的值
    * `thickness` 线的粗细

2. 方法二

    ```cpp
    std::vector<cv::Mat> drawRectangle(std::vector<cv::Mat> imgs,
        std::vector<torch::Tensor> rectangles, int thickness = 2)
    ```

    * `imgs` 需要绘制的图片集合
    * `rectangles` 预测的返回值
    * `thickness` 线的粗细

3. 方法三

    ```cpp
    std::vector<cv::Mat> drawRectangle(std::vector<cv::Mat> imgs, 
        std::vector<torch::Tensor> rectangles, 
        std::map<int, cv::Scalar> colors, 
        std::map<int, std::string> labels, int thickness = 2)
    ```

    * `imgs` 需要绘制的图片集合
    * `rectangles` 预测的返回值
    * `colors` 每种标签对应颜色
        * `key` 颜色编号从0开始与标签编号对应
        * `value` 颜色
    * `labels` 标签字典
        * `key` 标签编号从0开始
        * `value` 编号对应的值
    * `thickness` 线的粗细

4. 方法四

    ```cpp
    cv::Mat drawRectangle(cv::Mat img, torch::Tensor rectangle, int thickness = 2)
    ```

    * `img` 需要绘制的图片
    * `rectangle` 预测的返回值
    * `thickness` 线的粗细

5. 方法五

    ```cpp
    cv::Mat drawRectangle(cv::Mat img, torch::Tensor rectangle, std::map<int, std::string> labels, int thickness = 2)
    ```

    * `img` 需要绘制的图片
    * `rectangle` 预测的返回值
    * `labels` 标签字典
        * `key` 标签编号从0开始
        * `value` 编号对应的值
    * `thickness` 线的粗细

6. 方法六

    ```cpp
    cv::Mat drawRectangle(cv::Mat img, torch::Tensor rectangle, std::map<int, cv::Scalar> colors, std::map<int, std::string> labels, int thickness = 2)
    ```

    * `img` 需要绘制的图片
    * `rectangle` 预测的返回值
    * `colors` 每种标签对应颜色
        * `key` 颜色编号从0开始与标签编号对应
        * `value` 颜色
    * `labels` 标签字典
        * `key` 标签编号从0开始
        * `value` 编号对应的值
    * `thickness` 线的粗细
