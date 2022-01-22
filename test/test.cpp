#include <YoloV5.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
#include "YoloV5.h"

int main()
{
	// 第二个参数为是否启用 cuda 详细用法可以参考 YoloV5.h 文件
	YoloV5 yolo(torch::cuda::is_available() ? "./yolov5s.cuda.pt" : "./yolov5s.cpu.pt", torch::cuda::is_available());
	yolo.prediction(torch::rand({1, 3, 640, 640}));
	// 读取分类标签（我们用的官方的所以这里是 coco 中的分类）
	// 其实这些代码无所谓哪 只是后面预测出来的框没有标签罢了
	std::ifstream f("./coco.txt");
	std::string name = "";
	int i = 0;
	std::map<int, std::string> labels;
	while (std::getline(f, name))
	{
		labels.insert(std::pair<int, std::string>(i, name));
		i++;
	}
	// 用 OpenCV 打开摄像头读取文件（你随便咋样获取图片都OK哪）
	cv::VideoCapture cap = cv::VideoCapture(0);
	// 设置宽高 无所谓多宽多高后面都会通过一个算法转换为固定宽高的
	// 固定宽高值应该是你通过YoloV5训练得到的模型所需要的
	// 传入方式是构造 YoloV5 对象时传入 width 默认值为 640，height 默认值为 640
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 1000);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 800);
	cv::Mat frame;
	while (cap.isOpened())
	{
		// 读取一帧
		cap.read(frame);
		if (frame.empty())
		{
			std::cout << "Read frame failed!" << std::endl;
			break;
		}
		// 预测
		// 简单吧，两行代码预测结果就出来了，封装的还可以吧 嘚瑟
		clock_t start = clock();
		std::vector<torch::Tensor> r = yolo.prediction(frame);
		clock_t ends = clock();
		std::cout <<"Running Time : "<<(double)(ends - start) / CLOCKS_PER_SEC << std::endl;
		// 画框根据你自己的项目调用相应的方法，也可以不画框自己处理
		frame = yolo.drawRectangle(frame, r[0], labels);
		// show 图片
		cv::imshow("", frame);
		if (cv::waitKey(1) == 27) break;
	}
	return 0;
}