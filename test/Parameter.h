#pragma once
#include <iostream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

#define DEVICE "device"
#define OUTPUT "output"
#define ROI "roi"
#define WINDOW_WIDTH "window_width"
#define WINDOW_HEIGHT "window_height"
#define MODEL_WIDTH "model_width"
#define MODEL_HEIGHT "model_height"
#define ERROR_WIDTH 40
#define WINDOW_NAME "Yolo"
#define OUTPUT_SUFFIX ".mp4"
#define MODEL_PATH "model_path"
#define INPUT "input"
#define VERSION "version"
#define HELP "help"
#define IS_HALF "is_half"
#define STRING 1
#define INT 2
#define BOOL 4

void errorTemplate(std::string err);

class Desc
{
private:
	int d1;
	bool d2;
	std::string d3;
public:
	std::string name;
	std::string alias;
	std::string explain;
	int type;
	int t;
	Desc(std::string name, std::string alias, int data, int type, int t, std::string explain): name(name), alias(alias), d1(data), type(type), t(t), explain(explain) {};
	Desc(std::string name, std::string alias, bool data, int type, int t, std::string explain): name(name), alias(alias), d2(data), type(type), t(t), explain(explain) {};
	Desc(std::string name, std::string alias, std::string data, int type, int t, std::string explain): name(name), alias(alias), d3(data), type(type), t(t), explain(explain) {};
	int getI()
	{
		return d1;
	}
	bool getB()
	{
		return d2;
	}
	std::string getS()
	{
		return d3;
	}
	void set(int d, int t)
	{
		this->d1 = d;
		this->t = t;
	}
	void set(bool d, int t)
	{
		this->d2 = d;
		this->t = t;
	}
	void set(std::string d, int t)
	{
		this->d3 = d;
		this->t = t;
	}
};

class Parameter
{
private:	
	std::map<std::string, Desc> datas;
public:
	Parameter();
	void help();
	bool init(int argc, char const *argv[]);
	void print();
	Desc operator[](std::string key);
};