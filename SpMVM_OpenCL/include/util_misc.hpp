#include<compiler_config.h>

#ifndef UTIL_MISC_H
#define UTIL_MISC_H

#include<ctime>
#include<string>
#include<direct.h>
#include<stdlib.h>
#include<stdio.h>
#include<iostream>
#include<numeric> 

#include<CL/cl.h>
#include<JC/util.hpp>
#include<IO/convert_input.h>

std::string getTimeOfRun()
{
	time_t now = time(0);
	tm ltm;
	localtime_s(&ltm, &now);
	return "_" + std::to_string(ltm.tm_year + 1900) + std::to_string(ltm.tm_mon + 1) + std::to_string(ltm.tm_mday) + "_" + std::to_string(ltm.tm_hour) + (ltm.tm_min < 10 ? "0" : "") + std::to_string(ltm.tm_min) + (ltm.tm_sec < 10 ? "0" : "") + std::to_string(ltm.tm_sec);
}

double getMatrixDensity(unsigned long long matrix_n, IndexType matrix_nnz)
{
	return ((double)matrix_nnz / (matrix_n * matrix_n));
}

void printHeaderInfoSEQ(unsigned long long matrix_n, IndexType matrix_nnz)
{
	std::cout << "Matrix dimensions: " << matrix_n << std::endl << "Matrix non-zero element count: " << matrix_nnz << std::endl << "Matrix density: " << getMatrixDensity(matrix_n, matrix_nnz) << std::endl << "Operations count: " << 2 * matrix_nnz << std::endl << std::endl;
}

void printHeaderInfoGPU(unsigned long long matrix_n, IndexType matrix_nnz, std::string deviceName, std::string kernel_macros)
{
	std::cout << "Matrix dimensions: " << matrix_n << std::endl << "Matrix non-zero element count: " << matrix_nnz << std::endl << "Matrix density: " << getMatrixDensity(matrix_n, matrix_nnz) << std::endl << "Operations count: " << 2 * matrix_nnz << std::endl << "OpenCL device: " << deviceName << std::endl << "Kernel macros: " << kernel_macros << std::endl << std::endl;
}

void printRunInfo(unsigned long long repeat, unsigned long long nanoseconds, unsigned long long nnz, unsigned long long units_REAL, unsigned long long units_IndexType)
{
	std::cout << "Run: " << repeat << " | Time elapsed: " << nanoseconds << " ns | Effective throughput: " << (2 * nnz / (nanoseconds / 1e9)) / 1e9 << " GFLOPS | Effective bandwidth: " << ((units_REAL * sizeof(REAL)) + (units_IndexType * sizeof(IndexType))) / (nanoseconds / 1e9) / 1e9 << " GB/s\n";
}

void printAverageRunInfo(unsigned long long average_nanoseconds, unsigned long long nnz, unsigned long long units_REAL, unsigned long long units_IndexType)
{
	std::cout << std::endl << "Average time: " << average_nanoseconds << " ns | Average effective throughput: " << (2 * nnz / (average_nanoseconds / 1e9)) / 1e9 << " GFLOPS | Average effective bandwidth: " << ((units_REAL * sizeof(REAL)) + (units_IndexType * sizeof(IndexType))) / (average_nanoseconds / 1e9) / 1e9 << " GB/s\n";
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

void dumpPTXCode(cl::Program program, std::string filename) {
	// taken from URL: https://community.amd.com/thread/167373
	// Allocate some memory for all the kernel binary data  
	const std::vector<size_t> binSizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
	std::vector<char> binData(std::accumulate(binSizes.begin(), binSizes.end(), 0));
	char* binChunk = &binData[0];


	//A list of pointers to the binary data  
	std::vector<char*> binaries;
	for (unsigned int i = 0; i < binSizes.size(); ++i) {
		binaries.push_back(binChunk);
		binChunk += binSizes[i];
	}

	program.getInfo(CL_PROGRAM_BINARIES, &binaries[0]);
	std::ofstream binaryfile(filename, std::ios::binary);
	for (unsigned int i = 0; i < binaries.size(); ++i)
		binaryfile.write(binaries[i], binSizes[i]);
}

#endif