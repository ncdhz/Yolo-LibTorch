#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>

#define V6 "v6"
#define V8 "v8"
/**
 * ImageResizeData 图片处理过后保存图片的数据结构
 */
class ImageResizeData
{
public:
    // 设置处理过后的图片
	void setImg(cv::Mat img);
    // 获取处理过后的图片
	cv::Mat getImg();
    // 当原始图片宽高比大于处理过后图片宽高比时此函数返回 true
	bool isW();
    // 当原始图片高宽比大于处理过后图片高宽比时此函数返回 true
	bool isH();
    // 设置处理之后图片的宽
	void setWidth(int width);
    // 获取处理之后图片的宽
	int getWidth();
    // 设置处理之后图片的高
	void setHeight(int height);
    // 获取处理之后图片的高
	int getHeight();
    // 设置原始图片的宽
	void setW(int w);
    // 获取原始图片的宽
	int getW();
    // 设置原始图片的高
	void setH(int h);
    // 获取原始图片的高
	int getH();
    // 设置从原始图片到处理过后图片所添加黑边大小
	void setBorder(int border);
    // 获取从原始图片到处理过后图片所添加黑边大小
	int getBorder();
private:
    // 处理过后图片高
	int height;
	// 处理过后图片宽
    int width;
	// 原始图片宽
    int w;
	// 原始图片高
    int h;
	// 从原始图片到处理图片所添加的黑边大小
    int border;
	// 处理过后的图片
    cv::Mat img;
};

/**
 * Yolo 的实现类
 */
class Yolo
{
public:
    /**
     * 构造函数
     * @param ptFile Yolo pt文件路径
	 * @param version Yolo的版本 ["v5", "v6", "v7", "v8"] 中选一
	 * @param device 推理使用的设备默认为cpu
	 * @param height Yolo 训练时图片的高
	 * @param width Yolo 训练时图片的宽
	 * @param confThres 非极大值抑制中的 scoreThresh
	 * @param iouThres 非极大值抑制中的 iouThresh
     */
	Yolo(std::string ptFile, std::string version="v8", std::string device="cpu", bool isHalf=false, int height=640, int width=640,  float confThres=0.25, float iouThres=0.45);
	/**
	 * 预测函数
	 * @param data 需要预测的数据格式 (batch, rgb, height, width)
	 * @return std::vector 对应的是 batch size，torch::Tensor 的维度是 6 * n 表示（左上点x坐标，左上点y坐标，右下点x坐标，右下点y坐标，置信度，标签）n 表示有多少个框被预测出来
	 */
	std::vector<torch::Tensor> prediction(torch::Tensor data);
	/**
	 * 预测函数
	 * @param filePath 需要预测的图片路径
	 */
	std::vector<torch::Tensor> prediction(std::string filePath);
	/**
	 * 预测函数
	 * @param img 需要预测的图片
	 */
	std::vector<torch::Tensor> prediction(cv::Mat img);
	/**
	 * 预测函数 
	 * @param imgs 需要预测的图片集合
	 */
	std::vector<torch::Tensor> prediction(std::vector<cv::Mat> imgs);
	/**
	 * 改变图片大小的函数
	 * @param img 原始图片
	 * @param height 要处理成的图片的高
	 * @param width 要处理成的图片的宽
	 * @return 封装好的处理过后图片数据结构
	 */
	static ImageResizeData resize(cv::Mat img, int height, int width);
	/**
	 * 改变图片大小的函数
	 * @param img 原始图片
	 * @return 封装好的处理过后图片数据结构
	 */
	ImageResizeData resize(cv::Mat img);
	/**
	 * 改变图片大小的函数
	 * @param imgs 原始图片集合
	 * @param height 要处理成的图片的高
	 * @param width 要处理成的图片的宽
	 * @return 封装好的处理过后图片数据结构
	 */
	static std::vector<ImageResizeData> resize(std::vector<cv::Mat> imgs, int height, int width);
	/**
	 * 改变图片大小的函数
	 * @param imgs 原始图片集合
	 * @return 封装好的处理过后图片数据结构
	 */
	std::vector<ImageResizeData> resize(std::vector<cv::Mat> imgs);
	/**
	 * 根据输出结果在给定图片中画出框
	 * @param imgs 原始图片集合
	 * @param rectangles 通过预测函数处理好的结果
	 * @param labels 类别标签
	 * @param thickness 线宽
	 * @return 画好框的图片
	 */
	std::vector<cv::Mat> drawRectangle(std::vector<cv::Mat> imgs, std::vector<torch::Tensor> rectangles, std::map<int, std::string> labels, int thickness = 2);
	/**
	 * 根据输出结果在给定图片中画出框
	 * @param imgs 原始图片集合
	 * @param rectangles 通过预测函数处理好的结果
	 * @param thickness 线宽
	 * @return 画好框的图片
	 */
	std::vector<cv::Mat> drawRectangle(std::vector<cv::Mat> imgs, std::vector<torch::Tensor> rectangles, int thickness = 2);
	/**
	 * 根据输出结果在给定图片中画出框
	 * @param imgs 原始图片集合
	 * @param rectangles 通过预测函数处理好的结果
	 * @param colors 每种类型对应颜色
	 * @param labels 类别标签 
	 * @return 画好框的图片
	 */
	std::vector<cv::Mat> drawRectangle(std::vector<cv::Mat> imgs, std::vector<torch::Tensor> rectangles, std::map<int, cv::Scalar> colors, std::map<int, std::string> labels, int thickness = 2);
	/**
	 * 根据输出结果在给定图片中画出框
	 * @param img 原始图片
	 * @param rectangle 通过预测函数处理好的结果
	 * @param thickness 线宽
	 * @return 画好框的图片
	 */
	cv::Mat drawRectangle(cv::Mat img, torch::Tensor rectangle, int thickness = 2);
	/**
	 * 根据输出结果在给定图片中画出框
	 * @param img 原始图片
	 * @param rectangle 通过预测函数处理好的结果
	 * @param labels 类别标签
	 * @param thickness 线宽
	 * @return 画好框的图片
	 */
	cv::Mat drawRectangle(cv::Mat img, torch::Tensor rectangle, std::map<int, std::string> labels, int thickness = 2);
	/**
	 * 根据输出结果在给定图片中画出框
	 * @param img 原始图片
	 * @param rectangle 通过预测函数处理好的结果
	 * @param colos 每种类型对应颜色
	 * @param labels 类别标签
	 * @param thickness 线宽
	 * @return 画好框的图片
	 */
	cv::Mat drawRectangle(cv::Mat img, torch::Tensor rectangle, std::map<int, cv::Scalar> colors, std::map<int, std::string> labels, int thickness = 2);
	/**
	 * 用于判断给定数据是否存在预测
	 * @param clazz 通过预测函数处理好的结果
	 * @return 如果图片中存在给定某一种分类返回 true
	 */
	bool existencePrediction(torch::Tensor clazz);
	/**
	 * 用于判断给定数据是否存在预测
	 * @param classs 通过预测函数处理好的结果
	 * @return 如果图片集合中存在给定某一种分类返回 true
	 */
	bool existencePrediction(std::vector<torch::Tensor> classs);

private:
	// 推理使用的设备
	std::string device;
	// 是否使用半精度
	bool isHalf;
	// 非极大值抑制中的第一步数据清理
	float confThres;
	// 非极大值抑制中 iou
	float iouThres;
	// 模型所需要的图片的高
	float height;
	// 模型所需要的图片的宽
	float width;
	// Yolo 版本
	std::string version;
	// 画框颜色 map
	std::map<int, cv::Scalar> mainColors;
	// 模型
	torch::jit::script::Module model;
	// 随机获取一种颜色
	cv::Scalar getRandScalar();
	// 图片通道转换为 rgb
	cv::Mat img2RGB(cv::Mat img);
	// 图片变为 Tensor
	torch::Tensor img2Tensor(cv::Mat img);
	// (center_x center_y w h) to (left, top, right, bottom)
	torch::Tensor xywh2xyxy(torch::Tensor x);
	// 非极大值抑制算法
	torch::Tensor nms(torch::Tensor bboxes, torch::Tensor scores, float thresh);
	// 预测出来的框根据原始图片还原算法
	std::vector<torch::Tensor> sizeOriginal(std::vector<torch::Tensor> result, std::vector<ImageResizeData> imgRDs);
	// 非极大值抑制算法整体
	std::vector<torch::Tensor> non_max_suppression(torch::Tensor preds, float confThres = 0.25, float iouThres = 0.45);
};
