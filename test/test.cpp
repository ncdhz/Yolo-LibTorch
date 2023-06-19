#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
#include <map>
#include <vector>
#include "Yolo.h"
#include "signal.h"
#include "Parameter.h"

#define ROI_ON "on"
#define WINDOW_NAME "Yolo"
#define OUTPUT_SUFFIX ".mp4"

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

void roiHandle(Parameter para, cv::Mat &img, std::vector<cv::Point> points, std::vector<torch::Tensor> &r)
{
	if (para[ROI].t == STRING || para[ROI].getB())
	{
		cv::polylines(img, points, true, cv::Scalar(0, 0, 255));
		torch::Tensor preData = r.back();
		r.pop_back();
		if (para[ROI].t == BOOL && para[ROI].getB())
			r.push_back(getRegionData(points, preData, true));
		else
		{
			r.push_back(getRegionData(points, preData, !!(strcmp(para[ROI].getS().c_str(), ROI_ON) != 0)));
		}
	}
}

// 导出打标检测视频
cv::VideoWriter * outputVideo = nullptr;

void sigHandle(int sig)
{
	if (sig == SIGINT && outputVideo != nullptr)
	{
		outputVideo->release();
		delete outputVideo;
		exit(0);
	}
}

int main(int argc,  char const* argv[])
{
	signal(SIGINT, sigHandle);
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
	Yolo yolo(para[MODEL_PATH].getS(), para[VERSION].getS(), 
		para[DEVICE].getS(), para[IS_HALF].getB(), para[MODEL_HEIGHT].getI(), para[MODEL_WIDTH].getI());
	
	std::cout << "model loaded successfully." << std::endl;

	yolo.prediction(torch::rand({1, 3, para[MODEL_WIDTH].getI(), para[MODEL_HEIGHT].getI()}));

	// 读取分类标签（我们用的官方的所以这里是 coco 中的分类）
	// 其实这些代码无所谓哪 只是后面预测出来的框没有标签罢了
	std::ifstream f(para[LABEL_PATH].getS());
	std::string name = "";
	
	int i = 0;
	std::map<int, std::string> labels;
	while (std::getline(f, name))
	{
		labels.insert(std::pair<int, std::string>(i, name));
		i++;
	}

	if (!para[IS_CLOSE].getB())
	{
		cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL | cv::WINDOW_FREERATIO);
		cv::resizeWindow(WINDOW_NAME, para[WINDOW_WIDTH].getI(), para[WINDOW_HEIGHT].getI());
	}

	// image handle
	if (para[IS_IMAGE].getB())
	{
		std::string inputPath = para[INPUT].getS();
		cv::Mat img = cv::imread(inputPath);
		if (img.data == nullptr)
		{
			std::cout << "image path error." << std::endl;
		}
		else
		{
			std::vector<torch::Tensor> r = yolo.prediction(img);
			roiHandle(para, img, points, r);
			img = yolo.drawRectangle(img, r[0], labels);
			cv::imshow(WINDOW_NAME, img);
			cv::waitKey(0);
			if (para[OUTPUT].t == STRING)
			{
				cv::imwrite(para[OUTPUT].getS() + inputPath.substr(inputPath.find_last_of('.')), img);
			}
		}
		return 0;
	}

	// 用 OpenCV 打开摄像头或读取文件
	cv::VideoCapture *cap;
	if (para[INPUT].t == INT)
	{
		cap = new cv::VideoCapture(para[INPUT].getI());
	}
	else
	{
		cap = new cv::VideoCapture(para[INPUT].getS());
	}
	
	if (!cap->isOpened())
	{
		errorTemplate("input error");
		return 0;
	}

	// 设置宽高 无所谓多宽多高后面都会通过一个算法转换为固定宽高的
	// 固定宽高值应该是你通过Yolo训练得到的模型所需要的
	// 传入方式是构造 Yolo 对象时传入 width 默认值为 640，height 默认值为 640
	cap->set(cv::CAP_PROP_FRAME_WIDTH, para[WINDOW_WIDTH].getI());
	cap->set(cv::CAP_PROP_FRAME_HEIGHT, para[WINDOW_HEIGHT].getI());
	cv::Mat frame;

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
			int fps = cap->get(cv::CAP_PROP_FPS);
			if (!fps)
			{
				fps = 24;
			}
			outputVideo = new cv::VideoWriter(para[OUTPUT].getS() + OUTPUT_SUFFIX, cv::VideoWriter::fourcc('M', 'P', '4', 'V'), fps, frame.size());
		}

		// 预测
		clock_t start = clock();
		std::vector<torch::Tensor> r = yolo.prediction(frame);
		clock_t ends = clock();
		std::cout <<"Running Time : "<<(double)(ends - start) / CLOCKS_PER_SEC << std::endl;

		roiHandle(para, frame, points, r);

		// 画框根据你自己的项目调用相应的方法，也可以不画框自己处理
		frame = yolo.drawRectangle(frame, r[0], labels);
		
		if (outputVideo != nullptr)
		{
			outputVideo->write(frame);
		}
		// show and exit
		if (!para[IS_CLOSE].getB())
		{
			if (cv::waitKey(1) == 27 || cv::getWindowProperty(WINDOW_NAME, cv::WND_PROP_VISIBLE) < 1.0) break;
			cv::imshow(WINDOW_NAME, frame);
		}
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
