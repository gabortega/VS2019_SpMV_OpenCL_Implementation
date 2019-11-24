#ifndef UTIL_TIME_H
#define UTIL_TIME_H

#include<ctime>
#include<string>
#include<direct.h>
#include<stdlib.h>
#include<stdio.h>

std::string getTimeOfRun()
{
	time_t now = time(0);
	tm ltm;
	localtime_s(&ltm, &now);
	return "_" + std::to_string(ltm.tm_year + 1900) + std::to_string(ltm.tm_mon + 1) + std::to_string(ltm.tm_mday) + "_" + std::to_string(ltm.tm_hour) + (ltm.tm_min < 10 ? "0" : "") + std::to_string(ltm.tm_min) + (ltm.tm_sec < 10 ? "0" : "") + std::to_string(ltm.tm_sec);
}
#endif

int createOutputDirectory(std::string outputDirRoot, std::string outputDir) {
	int err;
	err = _mkdir(outputDirRoot.c_str());
	if (err == 0 || errno == EEXIST)
	{
		err = _mkdir((outputDirRoot + (std::string)"/" + outputDir).c_str());
		if (err == 0 || errno == EEXIST)
		{
			return 0;
		}
		else
			fprintf(stdout, "Problem creating output directory: %s", outputDir.c_str());
	}
	else
		fprintf(stdout, "Problem creating root output directory: %s", outputDirRoot.c_str());
	return 1;
}