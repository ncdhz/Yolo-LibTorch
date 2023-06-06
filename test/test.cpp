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
#define ROI_ON "on"
#define ROI_IN "in"
#define WIDTH "width"
#define HEIGHT "height"
#define PARA_NULL "null"
#define ERROR_WIDTH 30
#define WINDOW_NAME "Yolo"
#define OUTPUT_SUFFIX ".mp4"

char PARAMETERS[][3][10] = {
	{"-d", "--device", DEVICE},
	{"-o", "--output", OUTPUT},
	{"-r", "--roi", ROI},
	{"-x", "--width", WIDTH},
	{"-y", "--height", HEIGHT}
};

int PARAMETER_LEN = sizeof(PARAMETERS) / sizeof(PARAMETERS[0]);

void help()
{
	std::cout << "Usage: verification [options]" << std::endl << std::endl;
	std::cout << "Options:" << std::endl;
	std::cout << "-h    --help        understand the basic usage of software." << std::endl;
	std::cout << "-d    --device      please enter the device being detected, which can be a file or a camera. The default parameter is 0, indicating camera number 0." << std::endl;
	std::cout << "-o    --output      target detection data output file name." << std::endl;
	std::cout << "-r    --roi         enable region of interest detection. [on, in]" << std::endl;
	std::cout << "-x    --width       video display width." << std::endl;
	std::cout << "-y    --height      video display height." << std::endl;
	std::cout << std::endl;
}

bool initParameters(std::map<std::string, std::string>& paras, int argc, char const* argv[])
{
	paras[DEVICE] = "0";
	paras[OUTPUT] = PARA_NULL;
	paras[ROI] = PARA_NULL;
	paras[HEIGHT] = "640";
	paras[WIDTH] = "640";

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
					paras[PARAMETERS[j][2]] = ROI_IN;
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

void errorTemplate(std::string err)
{
	int center = (ERROR_WIDTH - err.length()) / 2 - 1;
	std::cout << std::string(ERROR_WIDTH, '-') << std::endl;
	std::cout << "|" << std::string(center, ' ');
	std::cout << err;
	std::cout << std::string(ERROR_WIDTH - err.length() - center - 2, ' ')  << "|" << std::endl;
	std::cout << std::string(ERROR_WIDTH, '-') << std::endl;
}

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
	std::map<std::string, std::string> paras;
	if (!initParameters(paras, argc, argv))
	{
		return 0;
	}
	for (int j = 0; j < PARAMETER_LEN; j++) {
		std::cout << std::left << std::setw(6) << PARAMETERS[j][0] << std::left << std::setw(15) << PARAMETERS[j][1] << paras[PARAMETERS[j][2]] << std::endl;
	}	
	// 初始化摄像头捕捉的范围，当输入为文件时可以忽略
	int width = 0;
	int height = 0;
	if ((height = atoi(paras[HEIGHT].c_str())) == 0)
	{
		errorTemplate("height error"); 
		return 0;
	}
	if ((width = atoi(paras[WIDTH].c_str())) == 0)
	{
		errorTemplate("width error"); 
		return 0;
	}

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
	cv::VideoCapture cap = cv::VideoCapture(paras[DEVICE]);
	if (!cap.isOpened())
	{
		errorTemplate("device error");
		return 0;
	}
	cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL | cv::WINDOW_FREERATIO);
	cv::resizeWindow(WINDOW_NAME, width, height);
	
	// 设置宽高 无所谓多宽多高后面都会通过一个算法转换为固定宽高的
	// 固定宽高值应该是你通过YoloV5训练得到的模型所需要的
	// 传入方式是构造 YoloV5 对象时传入 width 默认值为 640，height 默认值为 640
	cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
	cv::Mat frame;

	// 导出打标检测视频
	cv::VideoWriter * outputVideo = nullptr;

	while (cap.isOpened())
	{
		// 读取一帧
		cap.read(frame);
		if (frame.empty())
		{
			std::cout << "Read frame failed!" << std::endl;
			break;
		}
		// 初始化输出方法
		if (outputVideo == nullptr && strcmp(paras[OUTPUT].c_str(), PARA_NULL) != 0)
		{
			outputVideo = new cv::VideoWriter(paras[OUTPUT] + OUTPUT_SUFFIX, cv::VideoWriter::fourcc('M', 'P', '4', 'V'), cap.get(cv::CAP_PROP_FPS), frame.size());
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
			r.push_back(getRegionData(points, preData, !!strcmp(paras[ROI].c_str(), ROI_ON) != 0)); 
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

	cap.release();
	cv::destroyAllWindows();
	return 0;
}
