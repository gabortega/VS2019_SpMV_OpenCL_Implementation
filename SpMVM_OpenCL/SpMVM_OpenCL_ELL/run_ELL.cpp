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

#include"SpMVM_OpenCL_ELL.hpp"

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
#if ELL_SEQ || ELL || ELLG_SEQ || ELLG || HLL_SEQ || HLL
	// Error checking
	// TODO?

	FILE* f;
	struct coo_t coo;
	struct csr_t csr;
#if ELL_SEQ || ELL || ELLG_SEQ || ELLG
	struct ellg_t ellg;
#endif
#if HLL_SEQ || HLL
	struct hll_t hll;
#endif

#if INPUT_FILE_MODE
	std::string input_filename = (INPUT_FOLDER + (std::string)"/" + GENERATOR_FOLDER + (std::string)"/" + RANDOM_INPUT_FILE);
#else
	std::string input_filename = (INPUT_FOLDER + (std::string)"/" + INPUT_FILE);
#endif

	if (createOutputDirectory(OUTPUT_FOLDER, ELL_OUTPUT_FOLDER))
		exit(1);
	std::string output_file = (OUTPUT_FOLDER + (std::string)"/" + ELL_OUTPUT_FOLDER + (std::string)"/" + OUTPUT_FILENAME + getTimeOfRun() + OUTPUT_FILEFORMAT);

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
#if ELL_SEQ || ELL || ELLG_SEQ || ELLG
	if (!CSR_To_ELLG(&csr, &ellg, ELLG_LOG))
		std::cout << "ELL-G HAS BEEN TRUNCATED" << std::endl;
#endif
#if HLL_SEQ || HLL
	if (!CSR_To_HLL(&csr, &hll, HLL_LOG))
		std::cout << "HLL HAS BEEN TRUNCATED" << std::endl;
#endif
	FreeCOO(&coo);
	FreeCSR(&csr);
	std::cout << "-- DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;

	std::vector<CL_REAL> x = std::vector<CL_REAL>();
	for (IndexType i = 0; i < n; i++)
		x.push_back(i);

#if ELL_SEQ
	std::cout << std::endl << "-- STARTING ELL SEQUENTIAL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y1 = spmv_ELL_sequential(&ellg, x);
	std::cout << std::endl << "-- FINISHED ELL SEQUENTIAL OPERATION --" << std::endl << std::endl;
	if (ELL_SEQ_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y1.size(); i++)
			std::cout << y1[i] << " ";
		std::cout << std::endl;
	}
#endif
#if ELL
	std::cout << std::endl << "-- STARTING ELL KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y2 = spmv_ELL(&ellg, x);
	std::cout << std::endl << "-- FINISHED ELL KERNEL OPERATION --" << std::endl << std::endl;
	if (ELL_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y2.size(); i++)
			std::cout << y2[i] << " ";
		std::cout << std::endl;
	}
#endif
#if ELLG_SEQ
	std::cout << std::endl << "-- STARTING ELL-G SEQUENTIAL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y3 = spmv_ELLG_sequential(&ellg, x);
	std::cout << std::endl << "-- FINISHED ELL-G SEQUENTIAL OPERATION --" << std::endl << std::endl;
	if (ELLG_SEQ_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y3.size(); i++)
			std::cout << y3[i] << " ";
		std::cout << std::endl;
	}
#endif
#if ELLG
	std::cout << std::endl << "-- STARTING ELL-G KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y4 = spmv_ELLG(&ellg, x);
	std::cout << std::endl << "-- FINISHED ELL-G KERNEL OPERATION --" << std::endl << std::endl;
	if (ELLG_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y4.size(); i++)
			std::cout << y4[i] << " ";
		std::cout << std::endl;
	}
#endif
#if HLL_SEQ
	std::cout << std::endl << "-- STARTING HLL SEQUENTIAL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y5 = spmv_HLL_sequential(&hll, x);
	std::cout << std::endl << "-- FINISHED HLL SEQUENTIAL OPERATION --" << std::endl << std::endl;
	if (HLL_SEQ_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y5.size(); i++)
			std::cout << y5[i] << " ";
		std::cout << std::endl;
	}
#endif
#if HLL
	std::cout << std::endl << "-- STARTING HLL KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y6 = spmv_HLL(&hll, x);
	std::cout << std::endl << "-- FINISHED HLL KERNEL OPERATION --" << std::endl << std::endl;
	if (HLL_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y6.size(); i++)
			std::cout << y6[i] << " ";
		std::cout << std::endl;
	}
#endif

	x.clear();
#if ELL_SEQ || ELL || ELLG_SEQ || ELLG
	FreeELLG(&ellg);
#if ELL_SEQ
	y1.clear();
#endif
#if ELL
	y2.clear();
#endif
#if ELLG_SEQ
	y3.clear();
#endif
#if ELLG
	y4.clear();
#endif
#endif
#if HLL_SEQ || HLL
	FreeHLL(&hll);
#if HLL_SEQ
	y5.clear();
#endif
#if HLL
	y6.clear();
#endif
#endif
#endif
	return 0;
}