#include<compiler_config.h>

#include<iostream>
#include<stdio.h>
#include<string>
#include<vector>
#include<windows.h>

#include<util_misc.hpp>
#include<CL/cl.h>
#include<IO/mmio.h>
#include<IO/convert_input.h>

#include"SpMVM_OpenCL_CSR.hpp"

#if PRECISION == 2
#define CL_REAL cl_double
#else
//#elif PRECISION == 1
#define CL_REAL cl_float
//#else
//#define CL_REAL cl_half // TODO?
#endif

int main(void)
{
#if CSR_SEQ || CSR
	// Error checking
	// TODO?

	FILE* f;
	struct coo_t coo;
	struct csr_t csr;

#if INPUT_FILE_MODE
	std::string input_filename = (INPUT_FOLDER + (std::string)"/" + GENERATOR_FOLDER + (std::string)"/" + RANDOM_INPUT_FILE);
#else
	std::string input_filename = (INPUT_FOLDER + (std::string)"/" + INPUT_FILE);
#endif

	if (createOutputDirectory(OUTPUT_FOLDER, CSR_OUTPUT_FOLDER))
		exit(1);
	std::string output_file = (OUTPUT_FOLDER + (std::string)"/" + CSR_OUTPUT_FOLDER + (std::string)"/" + OUTPUT_FILENAME + getTimeOfRun() + OUTPUT_FILEFORMAT);

	std::cout << "!!! OUTPUT IS BEING WRITTEN TO " << output_file << " !!!" << std::endl;
	std::cout << "!!! PROGRAM WILL EXIT AUTOMATICALLY AFTER PROCESSING; PRESS CTRL-C TO ABORT !!!" << std::endl;
	system("PAUSE");
	freopen_s(&f, output_file.c_str(), "w", stdout);

	std::cout << "-- LOADING INPUT FILE " << input_filename << " --" << std::endl;
	MM_To_COO(input_filename.c_str(), &coo, COO_LOG);
	IndexType n = coo.n;
	std::cout << "-- INPUT FILE LOADED --" << std::endl << std::endl;
	std::cout << "-- PRE-PROCESSING INPUT --" << std::endl;
	COO_To_CSR(&coo, &csr, CSR_LOG);
	FreeCOO(&coo);
	std::cout << "-- DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;

	std::vector<CL_REAL> x = std::vector<CL_REAL>();
	for (IndexType i = 0; i < n; i++)
		x.push_back(i);

#if CSR_SEQ
	std::cout << std::endl << "-- STARTING CSR SEQUENTIAL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y1 = spmv_CSR_sequential(&csr, x);
	std::cout << std::endl << "-- FINISHED CSR SEQUENTIAL OPERATION --" << std::endl << std::endl;
	if (CSR_SEQ_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y1.size(); i++)
			std::cout << y1[i] << " ";
		std::cout << std::endl;
	}
#endif
#if CSR
	std::cout << std::endl << "-- STARTING CSR KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y2 = spmv_CSR(&csr, x);
	std::cout << std::endl << "-- FINISHED CSR KERNEL OPERATION --" << std::endl << std::endl;
	if (CSR_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y2.size(); i++)
			std::cout << y2[i] << " ";
		std::cout << std::endl;
	}
#endif

	x.clear();
	FreeCSR(&csr);
#if CSR_SEQ
	y1.clear();
#endif
#if CSR
	y2.clear();
#endif
#endif
	return 0;
}