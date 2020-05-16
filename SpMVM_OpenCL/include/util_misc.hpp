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

long double getCPWI(long double instr_count, unsigned long long nanoseconds)
{
	if (instr_count == 0 || nanoseconds == 0)
		return 0;
	else
	{
		// (runtime * clock frequency) / instr_count * CORE_COUNT * WARP_SIZE
		return ((nanoseconds / 1e9) * CORE_CLOCK_SPEED)/instr_count * CORE_COUNT * WARP_SIZE;
	}
}

std::string getGlobalConstants()
{
	return "-DPRECISION=" + std::to_string(PRECISION) + " -DUSE_CONSTANT_MEM=" + std::to_string(USE_CONSTANT_MEM);
}

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

// SEQ printing functions
void printHeaderInfoSEQ(unsigned long long matrix_n, IndexType matrix_nnz)
{
	std::cout << "Matrix dimensions: " << matrix_n << std::endl << "Matrix non-zero element count: " << matrix_nnz << std::endl << "Matrix density: " << getMatrixDensity(matrix_n, matrix_nnz) << std::endl << std::endl;
}

void printRunInfoSEQ(unsigned long long repeat, unsigned long long nanoseconds, unsigned long long nnz, unsigned long long units_REAL, unsigned long long units_IndexType)
{
	std::cout << "Run: " << repeat << " | Time elapsed: " << nanoseconds << " ns | Effective throughput: " << (2 * nnz / (nanoseconds / 1e9)) / 1e9 << " GFLOPS | Effective bandwidth: " << ((units_REAL * sizeof(REAL)) + (units_IndexType * sizeof(IndexType))) / (nanoseconds / 1e9) / 1e9 << " GB/s\n";
}

void printAverageRunInfoSEQ(unsigned long long average_nanoseconds, unsigned long long nnz, unsigned long long units_REAL, unsigned long long units_IndexType)
{
	std::cout << std::endl << "Average time: " << average_nanoseconds << " ns | Average effective throughput: " << (2 * nnz / (average_nanoseconds / 1e9)) / 1e9 << " GFLOPS | Average effective bandwidth: " << ((units_REAL * sizeof(REAL)) + (units_IndexType * sizeof(IndexType))) / (average_nanoseconds / 1e9) / 1e9 << " GB/s\n";
}

// GPU printing functions
void printHeaderInfoGPU(unsigned long long matrix_n, IndexType matrix_nnz, std::string deviceName, std::string kernel_macros, long double instr_count)
{
	std::cout << "Matrix dimensions: " << matrix_n << std::endl << "Matrix non-zero element count: " << matrix_nnz << std::endl << "Matrix density: " << getMatrixDensity(matrix_n, matrix_nnz) << std::endl << "OpenCL device: " << deviceName << std::endl << "Kernel macros: " << kernel_macros << std::endl << "Total kernel instructions: " << instr_count << std::endl << std::endl;
}

void printHeaderInfoGPU_HYB(unsigned long long matrix_n, IndexType matrix_nnz, std::string deviceName, std::string kernel_macros, long double instr_count_csr, long double instr_count_ell)
{
	std::cout << "Matrix dimensions: " << matrix_n << std::endl << "Matrix non-zero element count: " << matrix_nnz << std::endl << "Matrix density: " << getMatrixDensity(matrix_n, matrix_nnz) << std::endl << "OpenCL device: " << deviceName << std::endl << "Kernel macros: " << kernel_macros << std::endl << "Total kernel instructions: " << instr_count_csr + instr_count_ell << std::endl << "Total kernel (CSR) instructions: " << instr_count_csr << std::endl << "Total kernel (ELL) instructions: " << instr_count_ell << std::endl << std::endl;
}

void printRunInfoGPU(unsigned long long repeat, unsigned long long nanoseconds, unsigned long long nnz, unsigned long long units_REAL, unsigned long long units_IndexType, long double instr_count)
{
	std::cout << "Run: " << repeat << " | Time elapsed: " << nanoseconds << " ns | Effective throughput: " << (2 * nnz / (nanoseconds / 1e9)) / 1e9 << " GFLOPS | Effective bandwidth: " << ((units_REAL * sizeof(REAL)) + (units_IndexType * sizeof(IndexType))) / (nanoseconds / 1e9) / 1e9 << " GB/s | Effective CPWI per Core: " << getCPWI(instr_count, nanoseconds) << "\n";
}

void printRunInfoGPU_CSR(unsigned long long repeat, unsigned long long nanoseconds, unsigned long long nnz, IndexType coop, unsigned long long units_REAL, unsigned long long units_IndexType, long double instr_count)
{
	std::cout << "Run: " << repeat << " | Time elapsed: " << nanoseconds << " ns | Effective throughput: " << ((2 * nnz + max(1, log2(coop / 2))) / (nanoseconds / 1e9)) / 1e9 << " GFLOPS | Effective bandwidth: " << ((units_REAL * sizeof(REAL)) + (units_IndexType * sizeof(IndexType))) / (nanoseconds / 1e9) / 1e9 << " GB/s | Effective CPWI per Core: " << getCPWI(instr_count, nanoseconds) << "\n";
}

void printRunInfoGPU_HYB(unsigned long long repeat, unsigned long long nanoseconds_csr, unsigned long long nanoseconds_ell, unsigned long long nnz_csr, unsigned long long nnz_ell, IndexType coop, unsigned long long units_REAL, unsigned long long units_IndexType, long double instr_count_csr, long double instr_count_ell)
{
	std::cout << "Run: " << repeat << " | Time elapsed: " << (nanoseconds_csr + nanoseconds_ell) << " ns | Effective throughput: " << ((nnz_csr > 0) ? (((2 * nnz_csr + max(1, log2(coop / 2))) / ((nanoseconds_csr) / 1e9)) / 1e9) : 0) + ((nnz_ell > 0) ? (((2 * nnz_ell) / ((nanoseconds_ell) / 1e9)) / 1e9) : 0) << " GFLOPS | Effective bandwidth: " << ((units_REAL * sizeof(REAL)) + (units_IndexType * sizeof(IndexType))) / ((nanoseconds_csr + nanoseconds_ell) / 1e9) / 1e9 << " GB/s | Effective CPWI per Core: " << getCPWI(instr_count_csr, nanoseconds_csr) + getCPWI(instr_count_ell, nanoseconds_ell) << "\n";
}

void printAverageRunInfoGPU(unsigned long long average_nanoseconds, unsigned long long nnz, unsigned long long units_REAL, unsigned long long units_IndexType, long double instr_count)
{
	std::cout << std::endl << "Average time: " << average_nanoseconds << " ns | Average effective throughput: " << (2 * nnz / (average_nanoseconds / 1e9)) / 1e9 << " GFLOPS | Average effective bandwidth: " << ((units_REAL * sizeof(REAL)) + (units_IndexType * sizeof(IndexType))) / (average_nanoseconds / 1e9) / 1e9 << " GB/s | Average CPWI per Core: " << getCPWI(instr_count, average_nanoseconds) << "\n";
}

void printAverageRunInfoGPU_CSR(unsigned long long average_nanoseconds, unsigned long long nnz, IndexType coop, unsigned long long units_REAL, unsigned long long units_IndexType, long double instr_count)
{
	std::cout << std::endl << "Average time: " << average_nanoseconds << " ns | Average effective throughput: " << ((2 * nnz + max(1, log2(coop / 2))) / (average_nanoseconds / 1e9)) / 1e9 << " GFLOPS | Average effective bandwidth: " << ((units_REAL * sizeof(REAL)) + (units_IndexType * sizeof(IndexType))) / (average_nanoseconds / 1e9) / 1e9 << " GB/s | Average CPWI per Core: " << getCPWI(instr_count, average_nanoseconds) << "\n";
}

void printAverageRunInfoGPU_HYB(unsigned long long average_nanoseconds_csr, unsigned long long average_nanoseconds_ell, unsigned long long nnz_csr, unsigned long long nnz_ell, IndexType coop, unsigned long long units_REAL, unsigned long long units_IndexType, long double instr_count_csr, long double instr_count_ell)
{
	std::cout << std::endl << "Average time: " << (average_nanoseconds_csr + average_nanoseconds_ell) << " ns | Average effective throughput: " << ((nnz_csr > 0) ? (((2 * nnz_csr + max(1, log2(coop / 2))) / ((average_nanoseconds_csr) / 1e9)) / 1e9) : 0) + ((nnz_ell > 0) ? (((2 * nnz_ell) / ((average_nanoseconds_ell) / 1e9)) / 1e9) : 0) << " GFLOPS | Average effective bandwidth: " << ((units_REAL * sizeof(REAL)) + (units_IndexType * sizeof(IndexType))) / ((average_nanoseconds_csr + average_nanoseconds_ell) / 1e9) / 1e9 << " GB/s | Average CPWI per Core: " << getCPWI(instr_count_csr, average_nanoseconds_csr) + getCPWI(instr_count_ell, average_nanoseconds_ell) << "\n";
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

void dumpoBINCode(cl::Program program, std::string filename) {
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