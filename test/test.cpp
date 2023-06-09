#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
#include <iomanip>
#include <map>
#include <vector>
#include "Yolo.h"
#include "Parameter.h"

#define ROI_ON "on"

bool parameterError()
{
	if (std::cin.fail())
	{
		errorTemplate("parameter error");	
		return true;
	}
	return false;
}

bool roiParameters(std::vector<cv::Point>& points)
{
	std::cout << "please enter the number of points(which should be greater than or equal to 3): ";
	int num;
	std::cin >> num;
	if (parameterError())
	{
		return false;
	}
	if (num < 3) {
		std::cerr << "points must be greater than or equal to 3." << std::endl;
		return false;
	}
	for (size_t i = 1; i <= num; i++)
	{
		float x, y;
		std::cout << "please enter the x-coordinate of the " << i << " point : ";
		std::cin >> x;
		if (parameterError())
		{
			return false;
		}

		std::cout << "please enter the y-coordinate of the " << i << " point : ";
		std::cin >> y;
		if (parameterError())
		{
			return false;
		}

		points.push_back(cv::Point(x, y));
	}
	return true;
}

torch::Tensor getRegionData(std::vector<cv::Point> points, torch::Tensor preData, bool in)
{

	std::vector<int> index;

	for (int i = 0; i < preData.size(0); i++)
	{
		int sx = preData[i][0].item().toInt();
		int sy = preData[i][1].item().toInt();
		int ex = preData[i][2].item().toInt();
		int ey = preData[i][3].item().toInt();
			// 完全在里面
		if ((in && cv::pointPolygonTest(points, cv::Point(sx, sy), false) > 0 
			&& cv::pointPolygonTest(points, cv::Point(sx, ey), false) > 0 
			&& cv::pointPolygonTest(points, cv::Point(ex, sy), false) > 0 
			&& cv::pointPolygonTest(points, cv::Point(ex, ey), false) > 0) ||
			// 部分在里面
			(!in && (cv::pointPolygonTest(points, cv::Point(sx, sy), false) > 0 
			|| cv::pointPolygonTest(points, cv::Point(sx, ey), false) > 0 
			|| cv::pointPolygonTest(points, cv::Point(ex, sy), false) > 0 
			|| cv::pointPolygonTest(points, cv::Point(ex, ey), false) > 0))) 
		{
			index.push_back(i);
		}
	}
	return preData.index_select(0, torch::tensor(index));
}

int main(int argc,  char const* argv[])
{
	Parameter para;
	if (!para.init(argc, argv))
	{
		return 0;
	}
	if (para[HELP].getB())
	{
		para.help();
		return 0;
	}
	para.print();

	
	std::vector<cv::Point> points;

	if (para[ROI].t == STRING || para[ROI].getB()) {
		if (!roiParameters(points)) {
			return 0;
		}
	}

	// 第三个参数为是否启用 cuda 详细用法可以参考 Yolo.h 文件
	Yolo yolo(para[MODEL_PATH].getS(), para[VERSION].getS(), para[CUDA].getB());
	yolo.prediction(torch::rand({1, 3, para[MODEL_WIDTH].getI(), para[MODEL_HEIGHT].getI()}));
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
	
	// 用 OpenCV 打开摄像头或读取文件
	cv::VideoCapture *cap;
	if (para[DEVICE].t == INT)
	{
		cap = new cv::VideoCapture(para[DEVICE].getI());
	}
	else
	{
		cap = new cv::VideoCapture(para[DEVICE].getS());
	}
	
	if (!cap->isOpened())
	{
		errorTemplate("device error");
		return 0;
	}
	cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL | cv::WINDOW_FREERATIO);
	cv::resizeWindow(WINDOW_NAME, para[WINDOW_WIDTH].getI(), para[WINDOW_HEIGHT].getI());
	
	// 设置宽高 无所谓多宽多高后面都会通过一个算法转换为固定宽高的
	// 固定宽高值应该是你通过Yolo训练得到的模型所需要的
	// 传入方式是构造 Yolo 对象时传入 width 默认值为 640，height 默认值为 640
	cap->set(cv::CAP_PROP_FRAME_WIDTH, para[WINDOW_WIDTH].getI());
	cap->set(cv::CAP_PROP_FRAME_HEIGHT, para[WINDOW_HEIGHT].getI());
	cv::Mat frame;

	// 导出打标检测视频
	cv::VideoWriter * outputVideo = nullptr;

	while (cap->isOpened())
	{
		// 读取一帧
		cap->read(frame);
		if (frame.empty())
		{
			errorTemplate("read frame failed!");
			break;
		}
		// 初始化输出方法
		if (outputVideo == nullptr && para[OUTPUT].t == STRING)
		{
			outputVideo = new cv::VideoWriter(para[OUTPUT].getS() + OUTPUT_SUFFIX, cv::VideoWriter::fourcc('M', 'P', '4', 'V'), cap->get(cv::CAP_PROP_FPS), frame.size());
		}
		// 预测
		// 简单吧，两行代码预测结果就出来了，封装的还可以吧 嘚瑟
		clock_t start = clock();
		std::vector<torch::Tensor> r = yolo.prediction(frame);
		clock_t ends = clock();
		std::cout <<"Running Time : "<<(double)(ends - start) / CLOCKS_PER_SEC << std::endl;

		if (para[ROI].t == STRING || para[ROI].getB())
		{
			cv::polylines(frame, points, true, cv::Scalar(0, 0, 255));
			torch::Tensor preData = r.back();
			r.pop_back();
			if (para[ROI].t == BOOL && para[ROI].getB())
				r.push_back(getRegionData(points, preData, true));
			else
			{
				r.push_back(getRegionData(points, preData,!!(strcmp(para[ROI].getS().c_str(), ROI_ON) != 0)));
			}
			 
		}

		// 画框根据你自己的项目调用相应的方法，也可以不画框自己处理
		frame = yolo.drawRectangle(frame, r[0], labels);
		if (cv::waitKey(1) == 27 || cv::getWindowProperty(WINDOW_NAME, cv::WND_PROP_VISIBLE) < 1.0) break;

		if (outputVideo != nullptr)
		{
			outputVideo->write(frame);
		}
		// show 图片
		cv::imshow(WINDOW_NAME, frame);
	}

	if (outputVideo != nullptr)
	{
		outputVideo->release();
		delete outputVideo;
	}

	cap->release();
	delete cap;
	cv::destroyAllWindows();
	return 0;
}
