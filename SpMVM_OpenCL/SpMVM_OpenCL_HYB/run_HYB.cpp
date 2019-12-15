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

#include"SpMVM_OpenCL_HYB.hpp"

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
#if HYB_ELL_SEQ || HYB_ELL || HYB_ELLG_SEQ || HYB_ELLG || HYB_HLL_SEQ || HYB_HLL
	// Error checking
	// TODO?

	FILE* f;
	struct coo_t coo;
	struct csr_t csr;
#if HYB_ELL_SEQ || HYB_ELL || HYB_ELLG_SEQ || HYB_ELLG
	struct hybellg_t hyb_ellg;
#endif
#if HYB_HLL_SEQ || HYB_HLL
	struct hybhll_t hybhll_t;
#endif

#if INPUT_FILE_MODE
	std::string input_filename = (INPUT_FOLDER + (std::string)"/" + GENERATOR_FOLDER + (std::string)"/" + RANDOM_INPUT_FILE);
#else
	std::string input_filename = (INPUT_FOLDER + (std::string)"/" + INPUT_FILE);
#endif

	if (createOutputDirectory(OUTPUT_FOLDER, HYB_OUTPUT_FOLDER))
		exit(1);
	std::string output_file = (OUTPUT_FOLDER + (std::string)"/" + HYB_OUTPUT_FOLDER + (std::string)"/" + OUTPUT_FILENAME + getTimeOfRun() + OUTPUT_FILEFORMAT);

	std::cout << "!!! OUTPUT IS BEING WRITTEN TO " << output_file << " !!!" << std::endl;
	std::cout << "!!! PROGRAM WILL EXIT AUTOMATICALLY AFTER PROCESSING; PRESS CTRL-C TO ABORT !!!" << std::endl;
	system("PAUSE");
	freopen_s(&f, output_file.c_str(), "w", stdout);

	std::cout << "-- LOADING INPUT FILE " << input_filename << " --" << std::endl;
	MM_To_COO(input_filename.c_str(), &coo, COO_LOG);
	std::cout << "-- INPUT FILE LOADED --" << std::endl << std::endl;
	std::cout << "-- PRE-PROCESSING INPUT --" << std::endl;
	IndexType n = coo.n;
	COO_To_CSR(&coo, &csr, CSR_LOG);
	FreeCOO(&coo);
#if HYB_ELL_SEQ || HYB_ELL || HYB_ELLG_SEQ || HYB_ELLG
	CSR_To_HYBELLG(&csr, &hyb_ellg, HYB_ELLG_LOG);
#endif
#if HYB_HLL_SEQ || HYB_HLL
	CSR_To_HYBHLL(&csr, &hybhll_t, HYB_HLL_LOG);
#endif
	FreeCSR(&csr);
	std::cout << "-- DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;

	std::vector<CL_REAL> x = std::vector<CL_REAL>();
	for (IndexType i = 0; i < n; i++)
		x.push_back(i);

#if HYB_ELL_SEQ
	std::cout << std::endl << "-- STARTING HYB_ELL SEQUENTIAL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y1 = spmv_HYB_ELL_sequential(&hyb_ellg, x);
	std::cout << std::endl << "-- FINISHED HYB_ELL SEQUENTIAL OPERATION --" << std::endl << std::endl;
	if (HYB_ELL_SEQ_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y1.size(); i++)
			std::cout << y1[i] << " ";
		std::cout << std::endl;
	}
#endif
#if HYB_ELL
	std::cout << std::endl << "-- STARTING HYB_ELL KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y2 = spmv_HYB_ELL(&hyb_ellg, x);
	std::cout << std::endl << "-- FINISHED HYB_ELL KERNEL OPERATION --" << std::endl << std::endl;
	if (HYB_ELL_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y2.size(); i++)
			std::cout << y2[i] << " ";
		std::cout << std::endl;
	}
#endif
#if HYB_ELLG_SEQ
	std::cout << std::endl << "-- STARTING HYB_ELL-G SEQUENTIAL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y3 = spmv_HYB_ELLG_sequential(&hyb_ellg, x);
	std::cout << std::endl << "-- FINISHED HYB_ELL-G SEQUENTIAL OPERATION --" << std::endl << std::endl;
	if (HYB_ELLG_SEQ_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y3.size(); i++)
			std::cout << y3[i] << " ";
		std::cout << std::endl;
	}
#endif
#if HYB_ELLG
	std::cout << std::endl << "-- STARTING HYB_ELL-G KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y4 = spmv_HYB_ELLG(&hyb_ellg, x);
	std::cout << std::endl << "-- FINISHED HYB_ELL-G KERNEL OPERATION --" << std::endl << std::endl;
	if (HYB_ELLG_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y4.size(); i++)
			std::cout << y4[i] << " ";
		std::cout << std::endl;
	}
#endif
#if HYB_HLL_SEQ
	std::cout << std::endl << "-- STARTING HYB_HLL SEQUENTIAL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y5 = spmv_HYB_HLL_sequential(&hybhll_t, x);
	std::cout << std::endl << "-- FINISHED HYB_HLL SEQUENTIAL OPERATION --" << std::endl << std::endl;
	if (HYB_HLL_SEQ_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y5.size(); i++)
			std::cout << y5[i] << " ";
		std::cout << std::endl;
	}
#endif
#if HYB_HLL
	std::cout << std::endl << "-- STARTING HYB_HLL KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y6 = spmv_HYB_HLL(&hybhll_t, x);
	std::cout << std::endl << "-- FINISHED HYB_HLL KERNEL OPERATION --" << std::endl << std::endl;
	if (HYB_HLL_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y6.size(); i++)
			std::cout << y6[i] << " ";
		std::cout << std::endl;
	}
#endif

	x.clear();
#if HYB_ELL_SEQ || HYB_ELL || HYB_ELLG_SEQ || HYB_ELLG
	FreeHYBELLG(&hyb_ellg);
#if HYB_ELL_SEQ
	y1.clear();
#endif
#if HYB_ELL
	y2.clear();
#endif
#if HYB_ELLG_SEQ
	y3.clear();
#endif
#if HYB_ELLG
	y4.clear();
#endif
#endif
#if HYB_HLL_SEQ || HYB_HLL
	FreeHYBHLL(&hybhll_t);
#if HYB_HLL
	y5.clear();
#endif
#if HYB_HLL
	y6.clear();
#endif
#endif
#endif
	return 0;
}