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

#include"../SpMVM_OpenCL_CSR/SpMVM_OpenCL_CSR.hpp"
#include"../SpMVM_OpenCL_DIA/SpMVM_OpenCL_DIA.hpp"
#include"../SpMVM_OpenCL_ELL/SpMVM_OpenCL_ELL.hpp"
#include"../SpMVM_OpenCL_GMVM/SpMVM_OpenCL_GMVM.hpp"
#include"../SpMVM_OpenCL_HYB/SpMVM_OpenCL_HYB.hpp"
#include"../SpMVM_OpenCL_JAD/SpMVM_OpenCL_JAD.hpp"

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
    // Error checking
#if !(OVERRIDE_MEM) && (HDIA_OLD || ELLG_STRUCTS_NSEQ || HLL_STRUCTS_NSEQ || HYB_ELLG_STRUCTS_NSEQ || HYB_HLL_STRUCTS_NSEQ)
    std::cout << "!!! ERROR: OVERRIDE_MEM MUST BE SET TO 1 FOR OCCUPANCY TESTING !!!" << std::endl;
    system("PAUSE");
    exit(1);
#endif
#if COO_STRUCTS_NSEQ
#if DIA
    if (WORKGROUP_SIZE > MAX_NDIAG_PER_WG)
    {
        std::cout << "!!! ERROR: WORKGROUP_SIZE CANNOT BE GREATER THAN MAX_NDIAG_PER_WG !!!" << std::endl;
        system("PAUSE");
        exit(1);
    }
#endif
#if JAD
    if (WORKGROUP_SIZE > MAX_NJAD_PER_WG)
    {
        std::cout << "!!! ERROR: WORKGROUP_SIZE CANNOT BE GREATER THAN MAX_NJAD_PER_WG !!!" << std::endl;
        system("PAUSE");
        exit(1);
    }
#endif

    FILE* f;
    struct coo_t coo;
#if MAT_STRUCTS_NSEQ
    struct mat_t mat;
#endif
#if CSR_STRUCTS_NSEQ
    struct csr_t csr;
#if DIA_STRUCTS_NSEQ
    struct dia_t dia;
#endif
#if HDIA_STRUCTS_NSEQ
    struct hdia_t hdia;
#endif
#if ELLG_STRUCTS_NSEQ
    struct ellg_t ellg;
#endif
#if HLL_STRUCTS_NSEQ
    struct hll_t hll;
#endif
#if HYB_ELLG_STRUCTS_NSEQ
    struct hybellg_t hyb_ellg;
#endif
#if HYB_HLL_STRUCTS_NSEQ
    struct hybhll_t hybhll_t;
#endif
#if  JAD_STRUCTS_NSEQ
    struct jad_t jad;
#endif
#endif

#if INPUT_FILE_MODE
    std::string input_files = OCCUPANCY_RANDOM_INPUT_FILES;
#else
    std::string input_files = OCCUPANCY_INPUT_FILES;
#endif

    std::vector<std::string> input_filenames;

    size_t last = 0;
    size_t next = 0;
    while ((next = input_files.find(";", last)) != std::string::npos)
    {
        input_filenames.push_back(input_files.substr(last, next - last));
        last = next + 1;
    }
    input_filenames.push_back(input_files.substr(last));

    std::string input_filename;

    std::string output_file = (OUTPUT_FOLDER + (std::string)"/" + OCCUPANCY_OUTPUT_FOLDER + (std::string)"/" + OUTPUT_FILENAME + getTimeOfRun() + OUTPUT_FILEFORMAT);
    if (createOutputDirectory(OUTPUT_FOLDER, OCCUPANCY_OUTPUT_FOLDER))
        exit(1);
    std::cout << "!!! OUTPUT IS BEING WRITTEN TO " << output_file << " !!!" << std::endl;
    std::cout << "!!! PROGRAM WILL EXIT AUTOMATICALLY AFTER PROCESSING; PRESS CTRL-C TO ABORT !!!" << std::endl;
    system("PAUSE");
    freopen_s(&f, output_file.c_str(), "w", stdout);

    for (int run = 0; run < input_filenames.size(); run++)
    {
#if INPUT_FILE_MODE
        input_filename = (INPUT_FOLDER + (std::string)"/" + GENERATOR_FOLDER + (std::string)"/" + input_filenames[run]);
#else
        input_filename = (INPUT_FOLDER + (std::string)"/" + input_filenames[run]);
#endif

        std::cout << std::endl << "-- LOADING INPUT FILE " << input_filename << " --" << std::endl;
        MM_To_COO(input_filename.c_str(), &coo, COO_LOG);
        IndexType n = coo.n;

        std::vector<CL_REAL> x = std::vector<CL_REAL>();
        for (IndexType i = 0; i < n; i++)
            x.push_back(i);

        std::vector<CL_REAL> y;

        std::cout << "-- INPUT FILE LOADED --" << std::endl << std::endl;
        
        for (int wg = WARP_SIZE; wg <= MAX_WORKGROUP_SIZE; wg *= 2)
        {
            for (int conc_wgs = 1, prev_local_mem = -1, local_mem = MAX_LOCAL_MEM; conc_wgs <= MAX_CONC_WG; local_mem = MAX_LOCAL_MEM / (++conc_wgs))
            {
                local_mem -= (LOCAL_MEM_CHUNK_SIZE + (local_mem % LOCAL_MEM_CHUNK_SIZE));
                if (local_mem == prev_local_mem)
                    continue;
                prev_local_mem = local_mem;

                std::cout << "-- OCCUPANCY TEST PARAMETERS START --" << std::endl;
                std::cout << "OCCUPANCY_WORKGROUP_SIZE = " << wg << " threads" << std::endl;
                std::cout << "OCCUPANCY_LOCAL_MEM_SIZE = " << local_mem << " B" << std::endl;
                std::cout << "OCCUPANCY_THREAD_COUNT = " << MAX_OCCUPANCY_TEST_THREAD_COUNT << " threads" << std::endl;
                std::cout << "-- OCCUPANCY TEST PARAMETERS END --" << std::endl << std::endl;
                // -------------------------------------------- MAT struct
#if MAT_STRUCTS_NSEQ
                std::cout << "-- (GMVM) PRE-PROCESSING INPUT --" << std::endl;
                COO_To_MAT(&coo, &mat, MAT_LOG);
#if !(CSR_STRUCTS_NSEQ)
                FreeCOO(&coo);
#endif
                std::cout << "-- (GMVM) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
#if GMVM
                std::cout << std::endl << "-- STARTING GMVM KERNEL OPERATION --" << std::endl << std::endl;
                y = spmv_GMVM_param(&mat, x, wg, local_mem, MAX_OCCUPANCY_TEST_THREAD_COUNT);
                std::cout << std::endl << "-- FINISHED GMVM KERNEL OPERATION --" << std::endl << std::endl;
                if (GMVM_OUTPUT_LOG)
                {
                    std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
                    for (IndexType i = 0; i < y.size(); i++)
                        std::cout << y[i] << " ";
                    std::cout << std::endl;
                }
                y.clear();
                std::cout << std::endl;
#endif
                FreeMAT(&mat);
#endif
                // -------------------------------------------- CSR struct
#if CSR_STRUCTS_NSEQ
                std::cout << "-- (CSR) PRE-PROCESSING INPUT --" << std::endl;
                COO_To_CSR(&coo, &csr, CSR_LOG);
                std::cout << "-- (CSR) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
#if CSR
                std::cout << std::endl << "-- STARTING CSR KERNEL OPERATION --" << std::endl << std::endl;
                y = spmv_CSR_param(&csr, x, wg, local_mem, MAX_OCCUPANCY_TEST_THREAD_COUNT);
                std::cout << std::endl << "-- FINISHED CSR KERNEL OPERATION --" << std::endl << std::endl;
                if (CSR_OUTPUT_LOG)
                {
                    std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
                    for (IndexType i = 0; i < y.size(); i++)
                        std::cout << y[i] << " ";
                    std::cout << std::endl;
                }
                y.clear();
                std::cout << std::endl;
#endif
                // -------------------------------------------- DIA struct
#if DIA_STRUCTS_NSEQ
                std::cout << "-- (DIA) PRE-PROCESSING INPUT --" << std::endl;
                if (!CSR_To_DIA(&csr, &dia, DIA_LOG))
                    std::cout << "DIA IS INCOMPLETE" << std::endl;
                std::cout << "-- (DIA) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
#if DIA
                std::cout << std::endl << "-- STARTING DIA KERNEL OPERATION --" << std::endl << std::endl;
                y = spmv_DIA_param(&dia, x, wg, MAX_NDIAG_PER_WG, local_mem, MAX_OCCUPANCY_TEST_THREAD_COUNT);
                std::cout << std::endl << "-- FINISHED DIA KERNEL OPERATION --" << std::endl << std::endl;
                if (DIA_OUTPUT_LOG)
                {
                    std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
                    for (IndexType i = 0; i < y.size(); i++)
                        std::cout << y[i] << " ";
                    std::cout << std::endl;
                }
                std::cout << std::endl;
                y.clear();
                std::cout << std::endl;
#endif
#if TRANSPOSED_DIA
                transpose_DIA(&dia, TRANSPOSED_DIA_LOG);
                std::cout << std::endl << "-- STARTING TRANSPOSED DIA KERNEL OPERATION --" << std::endl << std::endl;
                std::vector<CL_REAL> y6 = spmv_TRANSPOSED_DIA_param(&dia, x, wg, MAX_NDIAG_PER_WG, local_mem, MAX_OCCUPANCY_TEST_THREAD_COUNT);
                std::cout << std::endl << "-- FINISHED TRANSPOSED DIA KERNEL OPERATION --" << std::endl << std::endl;
                if (TRANSPOSED_DIA_OUTPUT_LOG)
                {
                    std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
                    for (IndexType i = 0; i < y6.size(); i++)
                        std::cout << y6[i] << " ";
                    std::cout << std::endl;
                }
#endif
                FreeDIA(&dia);
#endif
                // -------------------------------------------- HDIA struct
#if HDIA_STRUCTS_NSEQ
                std::cout << "-- (HDIA) PRE-PROCESSING INPUT --" << std::endl;
                if (!CSR_To_HDIA(&csr, &hdia, HDIA_LOG))
                    std::cout << "HDIA IS INCOMPLETE" << std::endl;
                std::cout << "-- (HDIA) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
#if HDIA
                std::cout << std::endl << "-- STARTING HDIA KERNEL OPERATION --" << std::endl << std::endl;
                y = spmv_HDIA_param(&hdia, x, wg, MAX_NDIAG_PER_HACK, local_mem, MAX_OCCUPANCY_TEST_THREAD_COUNT);
                std::cout << std::endl << "-- FINISHED HDIA KERNEL OPERATION --" << std::endl << std::endl;
                if (HDIA_OUTPUT_LOG)
                {
                    std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
                    for (IndexType i = 0; i < y.size(); i++)
                        std::cout << y[i] << " ";
                }
                y.clear();
                std::cout << std::endl;
#endif
#if HDIA_OLD
                std::cout << std::endl << "-- STARTING HDIA (OLD) KERNEL OPERATION --" << std::endl << std::endl;
                y = spmv_HDIA_OLD_param(&hdia, x, wg, local_mem, MAX_OCCUPANCY_TEST_THREAD_COUNT);
                std::cout << std::endl << "-- FINISHED HDIA (OLD) KERNEL OPERATION --" << std::endl << std::endl;
                if (HDIA_OLD_OUTPUT_LOG)
                {
                    std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
                    for (IndexType i = 0; i < y.size(); i++)
                        std::cout << y[i] << " ";
                }
                y.clear();
                std::cout << std::endl;
#endif
                FreeHDIA(&hdia);
#endif
                // -------------------------------------------- ELLG struct
#if ELLG_STRUCTS_NSEQ
                std::cout << "-- (ELL-G) PRE-PROCESSING INPUT --" << std::endl;
                if (!CSR_To_ELLG(&csr, &ellg, ELLG_LOG))
                    std::cout << "ELL-G HAS BEEN TRUNCATED" << std::endl;
                std::cout << "-- (ELL-G) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
#if ELL
                std::cout << std::endl << "-- STARTING ELL KERNEL OPERATION --" << std::endl << std::endl;
                y = spmv_ELL_param(&ellg, x, wg, local_mem, MAX_OCCUPANCY_TEST_THREAD_COUNT);
                std::cout << std::endl << "-- FINISHED ELL KERNEL OPERATION --" << std::endl << std::endl;
                if (ELL_OUTPUT_LOG)
                {
                    std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
                    for (IndexType i = 0; i < y.size(); i++)
                        std::cout << y[i] << " ";
                    std::cout << std::endl;
                }
                y.clear();
                std::cout << std::endl;
#endif
#if ELLG
                std::cout << std::endl << "-- STARTING ELL-G KERNEL OPERATION --" << std::endl << std::endl;
                y = spmv_ELLG_param(&ellg, x, wg, local_mem, MAX_OCCUPANCY_TEST_THREAD_COUNT);
                std::cout << std::endl << "-- FINISHED ELL-G KERNEL OPERATION --" << std::endl << std::endl;
                if (ELLG_OUTPUT_LOG)
                {
                    std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
                    for (IndexType i = 0; i < y.size(); i++)
                        std::cout << y[i] << " ";
                    std::cout << std::endl;
                }
                y.clear();
                std::cout << std::endl;
#endif
#if TRANSPOSED_ELL || TRANSPOSED_ELLG
                transpose_ELLG(&ellg, TRANSPOSED_ELLG_LOG);
#if TRANSPOSED_ELL
                std::cout << std::endl << "-- STARTING TRANSPOSED ELL KERNEL OPERATION --" << std::endl << std::endl;
                y = spmv_TRANSPOSED_ELL_param(&ellg, x, wg, local_mem, MAX_OCCUPANCY_TEST_THREAD_COUNT);
                std::cout << std::endl << "-- FINISHED TRANSPOSED ELL KERNEL OPERATION --" << std::endl << std::endl;
                if (TRANSPOSED_ELL_OUTPUT_LOG)
                {
                    std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
                    for (IndexType i = 0; i < y.size(); i++)
                        std::cout << y[i] << " ";
                    std::cout << std::endl;
                }
#endif
#if TRANSPOSED_ELLG
                std::cout << std::endl << "-- STARTING TRANSPOSED ELL-G KERNEL OPERATION --" << std::endl << std::endl;
                y = spmv_TRANSPOSED_ELLG_param(&ellg, x, wg, local_mem, MAX_OCCUPANCY_TEST_THREAD_COUNT);
                std::cout << std::endl << "-- FINISHED TRANSPOSED ELL-G KERNEL OPERATION --" << std::endl << std::endl;
                if (TRANSPOSED_ELLG_OUTPUT_LOG)
                {
                    std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
                    for (IndexType i = 0; i < y.size(); i++)
                        std::cout << y[i] << " ";
                    std::cout << std::endl;
                }
#endif
#endif
                FreeELLG(&ellg);
#endif
                // -------------------------------------------- HLL struct
#if HLL_STRUCTS_NSEQ
                std::cout << "-- (HLL) PRE-PROCESSING INPUT --" << std::endl;
                if (!CSR_To_HLL(&csr, &hll, HLL_LOG))
                    std::cout << "HLL HAS BEEN TRUNCATED" << std::endl;
                std::cout << "-- (HLL) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
#if HLL
                std::cout << std::endl << "-- STARTING HLL KERNEL OPERATION --" << std::endl << std::endl;
                y = spmv_HLL_param(&hll, x, wg, local_mem, MAX_OCCUPANCY_TEST_THREAD_COUNT);
                std::cout << std::endl << "-- FINISHED HLL KERNEL OPERATION --" << std::endl << std::endl;
                if (HLL_OUTPUT_LOG)
                {
                    std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
                    for (IndexType i = 0; i < y.size(); i++)
                        std::cout << y[i] << " ";
                    std::cout << std::endl;
                }
                y.clear();
                std::cout << std::endl;
#endif
                FreeHLL(&hll);
#endif
                // -------------------------------------------- JAD struct
#if JAD_STRUCTS_NSEQ
                std::cout << "-- (JAD) PRE-PROCESSING INPUT --" << std::endl;
                CSR_To_JAD(&csr, &jad, JAD_LOG);
                std::cout << "-- (JAD) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
#if JAD
                std::cout << std::endl << "-- STARTING JAD KERNEL OPERATION --" << std::endl << std::endl;
                y = spmv_JAD_param(&jad, x, wg, MAX_NJAD_PER_WG, local_mem, MAX_OCCUPANCY_TEST_THREAD_COUNT);
                std::cout << std::endl << "-- FINISHED JAD KERNEL OPERATION --" << std::endl << std::endl;
                if (JAD_OUTPUT_LOG)
                {
                    std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
                    for (IndexType i = 0; i < y.size(); i++)
                        std::cout << y[i] << " ";
                    std::cout << std::endl;
                }
                y.clear();
                std::cout << std::endl;
#endif
                FreeJAD(&jad);
#endif
                FreeCSR(&csr);
#endif
            }
        }
        FreeCOO(&coo);
        x.clear();
    }
#endif
    exit(0);
}
