#ifndef UTIL_TIME_H
#define UTIL_TIME_H

#include<ctime>
#include<string>

std::string getTimeOfRun()
{
	time_t now = time(0);
	tm ltm;
	localtime_s(&ltm, &now);
	return "_" + std::to_string(ltm.tm_year + 1900) + std::to_string(ltm.tm_mon + 1) + std::to_string(ltm.tm_mday) + "_" + std::to_string(ltm.tm_hour) + (ltm.tm_min < 10 ? "0" : "") + std::to_string(ltm.tm_min) + (ltm.tm_sec < 10 ? "0" : "") + std::to_string(ltm.tm_sec);
}
#endif