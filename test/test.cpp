#include <YoloV5.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
	YoloV5 yolo("./yolov5s.torchscript.pt", false);
	std::ifstream f("./coco.txt");
	std::string name = "";
	int i = 0;
	std::map<int, std::string> labels;
	while (std::getline(f, name))
	{
		labels.insert(std::pair<int, std::string>(i, name));
		i++;
	}
	cv::VideoCapture cap = cv::VideoCapture(0);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
	cv::Mat frame;
	while (cap.isOpened())
	{
		clock_t start = clock();
		cap.read(frame);
		if (frame.empty())
		{
			std::cout << "Read frame failed!" << std::endl;
			break;
		}
		std::vector<torch::Tensor> r = yolo.prediction(frame);
		frame = yolo.drawRectangle(frame, r[0], labels);
		cv::imshow("", frame);
		if (cv::waitKey(1) == 27) break;
	}
	return	0;
}