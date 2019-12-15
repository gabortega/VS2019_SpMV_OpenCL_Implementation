#include<compiler_config.h>

#ifndef UTIL_MISC_H
#define UTIL_MISC_H

#include<ctime>
#include<string>
#include<direct.h>
#include<stdlib.h>
#include<stdio.h>
#include<iostream>

#include<IO/convert_input.h>

std::string getTimeOfRun()
{
	time_t now = time(0);
	tm ltm;
	localtime_s(&ltm, &now);
	return "_" + std::to_string(ltm.tm_year + 1900) + std::to_string(ltm.tm_mon + 1) + std::to_string(ltm.tm_mday) + "_" + std::to_string(ltm.tm_hour) + (ltm.tm_min < 10 ? "0" : "") + std::to_string(ltm.tm_min) + (ltm.tm_sec < 10 ? "0" : "") + std::to_string(ltm.tm_sec);
}

void printRunInfo(unsigned long long repeat, unsigned long long nanoseconds, unsigned long long nnz, unsigned long long units_REAL, unsigned long long units_IndexType)
{
	std::cout << "Run: " << repeat << " | Time elapsed: " << nanoseconds << " ns | Effective throughput: " << (2 * nnz / (nanoseconds / 1e9)) / 1e9 << "GFLOP/s | Effective bandwidth: " << ((units_REAL * sizeof(REAL)) + (units_IndexType * sizeof(IndexType))) / (nanoseconds / 1e9) / 1e9 << "GB/s\n";
}

void printAverageRunInfo(unsigned long long average_nanoseconds, unsigned long long nnz, unsigned long long units_REAL, unsigned long long units_IndexType)
{
	std::cout << std::endl << "Average time: " << average_nanoseconds << " ns | Average effective throughput: " << (2 * nnz / (average_nanoseconds / 1e9)) / 1e9 << "GFLOP/s | Average effective bandwidth: " << ((units_REAL * sizeof(REAL)) + (units_IndexType * sizeof(IndexType))) / (average_nanoseconds / 1e9) / 1e9 << "GB/s\n";
}

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

#endif