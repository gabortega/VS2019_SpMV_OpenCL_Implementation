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

#include"SpMVM_OpenCL_DIA.hpp"

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
#if DIA_SEQ || DIA || HDIA_SEQ || HDIA
	// Error checking
#if DIA
	if (WORKGROUP_SIZE > MAX_NDIAG_PER_WG)
	{
		std::cout << "!!! ERROR: WORKGROUP_SIZE CANNOT BE GREATER THAN MAX_NDIAG_PER_WG !!!" << std::endl;
		system("PAUSE");
		exit(1);
	}
#endif

	FILE* f;
	struct coo_t coo;
	struct csr_t csr;
#if DIA_SEQ || DIA
	struct dia_t dia;
#endif
#if HDIA_SEQ || HDIA
	struct hdia_t hdia;
#endif

#if INPUT_FILE_MODE
	std::string input_filename = (INPUT_FOLDER + (std::string)"/" + GENERATOR_FOLDER + (std::string)"/" + RANDOM_INPUT_FILE);
#else
	std::string input_filename = (INPUT_FOLDER + (std::string)"/" + INPUT_FILE);
#endif

	if (createOutputDirectory(OUTPUT_FOLDER, DIA_OUTPUT_FOLDER))
		exit(1);
	std::string output_file = (OUTPUT_FOLDER + (std::string)"/" + DIA_OUTPUT_FOLDER + (std::string)"/" + OUTPUT_FILENAME + getTimeOfRun() + OUTPUT_FILEFORMAT);

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
#if DIA_SEQ || DIA
	if (!CSR_To_DIA(&csr, &dia, DIA_LOG))
		std::cout << "DIA IS INCOMPLETE" << std::endl;
#endif
#if HDIA_SEQ || HDIA
	if (!CSR_To_HDIA(&csr, &hdia, HDIA_LOG))
		std::cout << "HDIA IS INCOMPLETE" << std::endl;
#endif
	FreeCOO(&coo);
	FreeCSR(&csr);
	std::cout << "-- DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;

	std::vector<CL_REAL> x = std::vector<CL_REAL>();
	for (IndexType i = 0; i < n; i++)
		x.push_back(i);

#if DIA_SEQ
	std::cout << std::endl << "-- STARTING DIA SEQUENTIAL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y1 = spmv_DIA_sequential(&dia, x);
	std::cout << std::endl << "-- FINISHED DIA SEQUENTIAL OPERATION --" << std::endl << std::endl;
	if (DIA_SEQ_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y1.size(); i++)
			std::cout << y1[i] << " ";
		std::cout << std::endl;
	}
#endif
#if DIA
	std::cout << std::endl << "-- STARTING DIA KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y2 = spmv_DIA(&dia, x);
	std::cout << std::endl << "-- FINISHED DIA KERNEL OPERATION --" << std::endl << std::endl;
	if (DIA_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y2.size(); i++)
			std::cout << y2[i] << " ";
		std::cout << std::endl;
	}
#endif
#if HDIA_SEQ
	std::cout << std::endl << "-- STARTING HDIA SEQUENTIAL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y3 = spmv_HDIA_sequential(&hdia, x);
	std::cout << std::endl << "-- FINISHED HDIA SEQUENTIAL OPERATION --" << std::endl << std::endl;
	if (HDIA_SEQ_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y3.size(); i++)
			std::cout << y3[i] << " ";
	}
	std::cout << std::endl;
#endif
#if HDIA
	std::cout << std::endl << "-- STARTING HDIA KERNEL OPERATION --" << std::endl << std::endl;
	std::vector<CL_REAL> y4 = spmv_HDIA(&hdia, x);
	std::cout << std::endl << "-- FINISHED HDIA KERNEL OPERATION --" << std::endl << std::endl;
	if (HDIA_OUTPUT_LOG)
	{
		std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
		for (IndexType i = 0; i < y4.size(); i++)
			std::cout << y4[i] << " ";
	}
	std::cout << std::endl;
#endif

	x.clear();
#if DIA_SEQ || DIA
	FreeDIA(&dia);
#if DIA_SEQ
	y1.clear();
#endif
#if DIA
	y2.clear();
#endif
#endif
#if HDIA_SEQ || HDIA
	FreeHDIA(&hdia);
#if HDIA_SEQ
	y3.clear();
#endif
#if HDIA
	y4.clear();
#endif
#endif
#endif
	return 0;
}