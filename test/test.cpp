#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
#include <iomanip>
#include <map>
#include <vector>
#include "YoloV5.h"

#define DEVICE "device"
#define OUTPUT "output"
#define ROI "roi"
#define PARA_NULL "null"
#define PARA_TRUE "true"

char PARAMETERS[][3][10] = {
	{"-d", "--device", DEVICE},
	{"-o", "--output", OUTPUT},
	{"-r", "--roi", ROI}
};

int PARAMETER_LEN = sizeof(PARAMETERS) / sizeof(PARAMETERS[0]);

void help()
{
	std::cout << "Usage: verification [options]" << std::endl << std::endl;
	std::cout << "Options:" << std::endl;
	std::cout << "-h    --help        understand the basic usage of software." << std::endl;
	std::cout << "-d    --device      please enter the device being detected, which can be a file or a camera. The default parameter is 0, indicating camera number 0." << std::endl;
	std::cout << "-o    --output      target detection data output path." << std::endl;
	std::cout << "-r    --roi         enable region of interest detection." << std::endl;
	std::cout << std::endl;
}

bool initParameters(std::map<std::string, std::string>& paras, int argc, char const* argv[])
{
	paras[DEVICE] = "0";
	paras[OUTPUT] = PARA_NULL;
	paras[ROI] = PARA_NULL;

	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
		{
			help();
			return false;
		}
		for (int j = 0; j < PARAMETER_LEN; j++) {
			if (strcmp(argv[i], PARAMETERS[j][0]) == 0 || strcmp(argv[i], PARAMETERS[j][1]) == 0)
			{
				if (strcmp(PARAMETERS[j][2], ROI) == 0) {
					paras[PARAMETERS[j][2]] = PARA_TRUE;
				}

				if (i + 1 < argc)
				{
					if (strcmp(PARAMETERS[j][2], ROI) == 0 && argv[i + 1][0] == '-') {
						break;
					}
					std::string str(argv[i + 1]);
					paras[PARAMETERS[j][2]] = str;
					i++;
				}
				else
				{
					if (strcmp(PARAMETERS[j][2], ROI) != 0) {
						std::cout << "Incomplete parameters." << std::endl;
						return false;
					}
				}
				break;
			}
		}
	}
	return true;
}

bool parameterError()
{
	if (std::cin.fail())
	{	
		std::cout << "---------------------" << std::endl;
		std::cout << "|  parameter error  |" << std::endl;
		std::cout << "---------------------" << std::endl;
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
		std::cout << "points must be greater than or equal to 3." << std::endl;
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

torch::Tensor getRegionData(std::vector<cv::Point> points, torch::Tensor preData)
{

	std::vector<int> index;

	for (int i = 0; i < preData.size(0); i++)
	{
		int sx = preData[i][0].item().toInt();
		int sy = preData[i][1].item().toInt();
		int ex = preData[i][2].item().toInt();
		int ey = preData[i][3].item().toInt();
		if (cv::pointPolygonTest(points, cv::Point(sx, sy), false) > 0 || cv::pointPolygonTest(points, cv::Point(sx, ey), false) > 0 
			|| cv::pointPolygonTest(points, cv::Point(ex, sy), false) > 0 || cv::pointPolygonTest(points, cv::Point(ex, ey), false) > 0) 
		{
			index.push_back(i);
		}
	}
	return preData.index_select(0, torch::tensor(index));
}

int main(int argc,  char const* argv[])
{
	std::map<std::string, std::string> paras;
	if (!initParameters(paras, argc, argv))
	{
		return 0;
	}
	for (int j = 0; j < PARAMETER_LEN; j++) {
		std::cout << std::left << std::setw(6) << PARAMETERS[j][0] << std::left << std::setw(15) << PARAMETERS[j][1] << paras[PARAMETERS[j][2]] << std::endl;
	}
	std::cout << std::endl;

	bool existRoi = false;
	
	std::vector<cv::Point> points;

	if (strcmp(paras[ROI].c_str(), PARA_NULL) != 0) {
		existRoi = true;
		if (!roiParameters(points)) {
			return 0;
		}
	}

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

		if (existRoi)
		{
			cv::polylines(frame, points, true, cv::Scalar(0, 0, 255));
			torch::Tensor preData = r.back();
			r.pop_back();
			r.push_back(getRegionData(points, preData)); 
		}

		// 画框根据你自己的项目调用相应的方法，也可以不画框自己处理
		frame = yolo.drawRectangle(frame, r[0], labels);
		// show 图片
		cv::imshow("", frame);
		if (cv::waitKey(1) == 27) break;
	}
	return 0;
}
