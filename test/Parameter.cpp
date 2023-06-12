#include "Parameter.h"

Parameter::Parameter()
{
	this->datas.insert(std::make_pair(HELP, Desc("--help", "-h", false, BOOL, BOOL, "understand the basic usage of software.")));
	this->datas.insert(std::make_pair(INPUT, Desc("--input", "-i", 0, INT | STRING, INT, 
		"please enter the device being detected, which can be a file or a camera. default 0.")));
	this->datas.insert(std::make_pair(OUTPUT, Desc("--output", "-o", false, BOOL | STRING, BOOL, "target detection data output file name.")));
	this->datas.insert(std::make_pair(ROI, Desc("--roi", "-r", false, STRING | BOOL, BOOL, "enable region of interest detection. [on, in]")));
	this->datas.insert(std::make_pair(DEVICE, Desc("--device", "-d", std::string("cpu"), STRING, STRING, "cuda device, i.e. cuda:0 or cuda:1,cuda:2 or cpu")));
	this->datas.insert(std::make_pair(VERSION, Desc("--version", "-v", std::string("v8"), STRING, STRING, "using the version of the model. [v5, v6, v7, v8]")));

	this->datas.insert(std::make_pair(IS_HALF, Desc("--is_half", "-ih", false, BOOL, BOOL, "open half precision.")));
	this->datas.insert(std::make_pair(WINDOW_HEIGHT, Desc("--window_height", "-wh", 640, INT, INT, "window display height.")));
	this->datas.insert(std::make_pair(WINDOW_WIDTH, Desc("--window_width", "-ww", 640, INT, INT, "window display width.")));
	this->datas.insert(std::make_pair(MODEL_PATH, Desc("--model_path", "-mp", std::string("yolov8n.cpu.torchscript"), STRING, STRING, "model path.")));
	this->datas.insert(std::make_pair(MODEL_WIDTH, Desc("--model_width", "-mw", 640, INT, INT, "model input width.")));
	this->datas.insert(std::make_pair(MODEL_HEIGHT, Desc("--model_height", "-mh", 640, INT, INT, "model input height.")));
}

void Parameter::help()
{
	std::cout << "Usage: verification [options]" << std::endl;
	std::cout << "Options:" << std::endl;
	auto iter = this->datas.begin();
	while (iter != this->datas.end())
	{
		Desc desc = iter->second;
		std::cout << std::left << std::setw(5) << desc.alias;
		std::cout << std::left << std::setw(20) << desc.name;
		std::cout << desc.explain << std::endl;
		iter ++;
	}
}

void Parameter::print()
{
	auto iter = this->datas.begin();
	while (iter != this->datas.end())
	{
		Desc desc = iter->second;
		std::cout << std::left << std::setw(5) << desc.alias;
		std::cout << std::left << std::setw(20) << desc.name;
		
		if (desc.t == INT)
		{
			std::cout << desc.getI();
		}
		else if (desc.t == STRING)
		{
			std::cout << desc.getS();
		}
		else
		{
			std::cout << desc.getB();
		}
		std::cout << std::endl;
		iter ++;
	}
}

bool isNum(std::string str)
{
	for (int i = 0; i < str.length(); i++)
	{
		if (std::isdigit(str.c_str()[i]) == 0) return false;
	}
	return true;
}

bool Parameter::init(int argc, char const* argv[])
{
	for (int i = 1; i < argc; i++)
	{
		if (argv[i][0] != '-')
		{
			errorTemplate("parameter error");
			return false;
		}
		auto iter = this->datas.begin();
		while (iter != this->datas.end())
		{
			Desc& desc = iter->second;
			iter++;
			if (strcmp(argv[i], desc.name.c_str()) == 0 || strcmp(argv[i], desc.alias.c_str()) == 0)
			{
				if ((desc.type & BOOL) == BOOL)
				{
					desc.set(true, BOOL);
					if ((i + 1 < argc && argv[i + 1][0] == '-') || i + 1 >= argc)
					{
						break;
					}
				}
				if (i + 1 >= argc)
				{
					errorTemplate("parameter error [" + std::string(argv[i]) + "]");
					return false;
				}
				if ((desc.type & INT) == INT)
				{
					if (isNum(std::string(argv[i + 1])))
					{

						desc.set(atoi(argv[++i]), INT);
						break;
					}
				}
				if ((desc.type & STRING) == STRING)
				{
					desc.set(std::string(argv[++i]), STRING);
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

Desc Parameter::operator[](std::string key)
{
	return this->datas.find(key)->second;
}
