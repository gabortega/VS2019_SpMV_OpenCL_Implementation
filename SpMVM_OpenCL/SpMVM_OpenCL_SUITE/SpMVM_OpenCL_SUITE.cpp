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
#if CSR || DIA_SEQ || DIA || HDIA_SEQ || HDIA || ELL_SEQ || ELL || ELLG || HLL_SEQ || HLL || GMVM_SEQ || GMVM || HYB_ELL_SEQ || HYB_ELL || HYB_ELLG_SEQ || HYB_ELLG || HYB_HLL_SEQ || HYB_HLL || JAD_SEQ || JAD
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
#if GMVM_SEQ || GMVM
    struct mat_t mat;
#endif
#if CSR_SEQ || CSR || DIA_SEQ || DIA || HDIA_SEQ || HDIA ||ELL_SEQ || ELL || ELLG_SEQ || ELLG || HLL_SEQ || HLL || HYB_ELL_SEQ || HYB_ELL || HYB_ELLG_SEQ || HYB_ELLG || HYB_HLL_SEQ || HYB_HLL || JAD_SEQ || JAD
    struct csr_t csr;
#if DIA_SEQ || DIA
    struct dia_t dia;
#endif
#if HDIA_SEQ || HDIA || HDIA_OLD
    struct hdia_t hdia;
#endif
#if ELL_SEQ || ELL || ELLG_SEQ || ELLG
    struct ellg_t ellg;
#endif
#if HLL_SEQ || HLL
    struct hll_t hll;
#endif
#if HYB_ELL_SEQ || HYB_ELL || HYB_ELLG_SEQ || HYB_ELLG
    struct hybellg_t hyb_ellg;
#endif
#if HYB_HLL_SEQ || HYB_HLL
    struct hybhll_t hybhll_t;
#endif
#if  JAD_SEQ || JAD
    struct jad_t jad;
#endif
#endif

#if INPUT_FILE_MODE
    std::string input_files = SUITE_RANDOM_INPUT_FILES;
#else
    std::string input_files = SUITE_INPUT_FILES;
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

    std::string output_file = (OUTPUT_FOLDER + (std::string)"/" + SUITE_OUTPUT_FOLDER + (std::string)"/" + OUTPUT_FILENAME + getTimeOfRun() + OUTPUT_FILEFORMAT);
    if (createOutputDirectory(OUTPUT_FOLDER, SUITE_OUTPUT_FOLDER))
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

        // -------------------------------------------- MAT struct
#if GMVM_SEQ || GMVM
        std::cout << "-- (GMVM) PRE-PROCESSING INPUT --" << std::endl;
        COO_To_MAT(&coo, &mat, MAT_LOG);
#if !(CSR || DIA_SEQ || DIA || HDIA_SEQ || HDIA || ELL_SEQ || ELL || ELLG || HLL_SEQ || HLL || HYB_ELL_SEQ || HYB_ELL || HYB_ELLG_SEQ || HYB_ELLG || HYB_HLL_SEQ || HYB_HLL || JAD_SEQ || JAD)
        FreeCOO(&coo);
#endif
        std::cout << "-- (GMVM) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
#if GMVM_SEQ
        std::cout << std::endl << "-- STARTING GMVM SEQUENTIAL OPERATION --" << std::endl << std::endl;
        y = spmv_GMVM_sequential(&mat, x);
        std::cout << std::endl << "-- FINISHED GMVM SEQUENTIAL OPERATION --" << std::endl << std::endl;
        if (GMVM_SEQ_OUTPUT_LOG)
        {
            std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
            for (IndexType i = 0; i < y.size(); i++)
                std::cout << y[i] << " ";
            std::cout << std::endl;
        }
        y.clear();
        std::cout << std::endl;
#endif
#if GMVM
        std::cout << std::endl << "-- STARTING GMVM KERNEL OPERATION --" << std::endl << std::endl;
        y = spmv_GMVM(&mat, x);
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
#if CSR || DIA_SEQ || DIA || TRANSPOSED_DIA || HDIA_SEQ || HDIA || ELL_SEQ || ELL || TRANSPOSED_ELL || ELLG_SEQ || ELLG || TRANSPOSED_ELLG || HLL_SEQ || HLL || HYB_ELL_SEQ || HYB_ELL || HYB_ELLG_SEQ || HYB_ELLG || HYB_HLL_SEQ || HYB_HLL || JAD_SEQ || JAD
        std::cout << "-- (CSR) PRE-PROCESSING INPUT --" << std::endl;
        COO_To_CSR(&coo, &csr, CSR_LOG);
        FreeCOO(&coo);
        std::cout << "-- (CSR) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
#if CSR_SEQ
        std::cout << std::endl << "-- STARTING CSR SEQUENTIAL OPERATION --" << std::endl << std::endl;
        y = spmv_CSR_sequential(&csr, x);
        std::cout << std::endl << "-- FINISHED CSR SEQUENTIAL OPERATION --" << std::endl << std::endl;
        if (CSR_SEQ_OUTPUT_LOG)
        {
            std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
            for (IndexType i = 0; i < y.size(); i++)
                std::cout << y[i] << " ";
            std::cout << std::endl;
        }
        y.clear();
        std::cout << std::endl;
#endif
#if CSR
        std::cout << std::endl << "-- STARTING CSR KERNEL OPERATION --" << std::endl << std::endl;
        y = spmv_CSR(&csr, x);
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
#if DIA_SEQ || DIA || TRANSPOSED_DIA
        std::cout << "-- (DIA) PRE-PROCESSING INPUT --" << std::endl;
        if (!CSR_To_DIA(&csr, &dia, DIA_LOG))
            std::cout << "DIA IS INCOMPLETE" << std::endl;
        std::cout << "-- (DIA) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
#if DIA_SEQ
        std::cout << std::endl << "-- STARTING DIA SEQUENTIAL OPERATION --" << std::endl << std::endl;
        y = spmv_DIA_sequential(&dia, x);
        std::cout << std::endl << "-- FINISHED DIA SEQUENTIAL OPERATION --" << std::endl << std::endl;
        if (DIA_SEQ_OUTPUT_LOG)
        {
            std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
            for (IndexType i = 0; i < y.size(); i++)
                std::cout << y[i] << " ";
            std::cout << std::endl;
        }
        y.clear();
        std::cout << std::endl;
#endif
#if DIA
        std::cout << std::endl << "-- STARTING DIA KERNEL OPERATION --" << std::endl << std::endl;
        y = spmv_DIA(&dia, x);
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
        std::vector<CL_REAL> y6 = spmv_TRANSPOSED_DIA(&dia, x);
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
#if HDIA_SEQ || HDIA || HDIA_OLD
        std::cout << "-- (HDIA) PRE-PROCESSING INPUT --" << std::endl;
        if (!CSR_To_HDIA(&csr, &hdia, HDIA_LOG))
            std::cout << "HDIA IS INCOMPLETE" << std::endl;
        std::cout << "-- (HDIA) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
#if HDIA_SEQ
        std::cout << std::endl << "-- STARTING HDIA SEQUENTIAL OPERATION --" << std::endl << std::endl;
        y = spmv_HDIA_sequential(&hdia, x);
        std::cout << std::endl << "-- FINISHED HDIA SEQUENTIAL OPERATION --" << std::endl << std::endl;
        if (HDIA_SEQ_OUTPUT_LOG)
        {
            std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
            for (IndexType i = 0; i < y.size(); i++)
                std::cout << y[i] << " ";
        }
        y.clear();
        std::cout << std::endl;
#endif
#if HDIA
        std::cout << std::endl << "-- STARTING HDIA KERNEL OPERATION --" << std::endl << std::endl;
        y = spmv_HDIA(&hdia, x);
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
        y = spmv_HDIA_OLD(&hdia, x);
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
#if ELL_SEQ || ELL || TRANSPOSED_ELL || ELLG_SEQ || ELLG || TRANSPOSED_ELLG
        std::cout << "-- (ELL-G) PRE-PROCESSING INPUT --" << std::endl;
        if (!CSR_To_ELLG(&csr, &ellg, ELLG_LOG))
            std::cout << "ELL-G HAS BEEN TRUNCATED" << std::endl;
        std::cout << "-- (ELL-G) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
#if ELL_SEQ
        std::cout << std::endl << "-- STARTING ELL SEQUENTIAL OPERATION --" << std::endl << std::endl;
        y = spmv_ELL_sequential(&ellg, x);
        std::cout << std::endl << "-- FINISHED ELL SEQUENTIAL OPERATION --" << std::endl << std::endl;
        if (ELL_SEQ_OUTPUT_LOG)
        {
            std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
            for (IndexType i = 0; i < y.size(); i++)
                std::cout << y[i] << " ";
            std::cout << std::endl;
        }
        y.clear();
        std::cout << std::endl;
#endif
#if ELL
        std::cout << std::endl << "-- STARTING ELL KERNEL OPERATION --" << std::endl << std::endl;
        y = spmv_ELL(&ellg, x);
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
#if ELLG_SEQ
        std::cout << std::endl << "-- STARTING ELL-G SEQUENTIAL OPERATION --" << std::endl << std::endl;
        y = spmv_ELLG_sequential(&ellg, x);
        std::cout << std::endl << "-- FINISHED ELL-G SEQUENTIAL OPERATION --" << std::endl << std::endl;
        if (ELLG_SEQ_OUTPUT_LOG)
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
        y = spmv_ELLG(&ellg, x);
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
        y = spmv_TRANSPOSED_ELL(&ellg, x);
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
        y = spmv_TRANSPOSED_ELLG(&ellg, x);
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
#if HLL_SEQ || HLL
        std::cout << "-- (HLL) PRE-PROCESSING INPUT --" << std::endl;
        if (!CSR_To_HLL(&csr, &hll, HLL_LOG))
            std::cout << "HLL HAS BEEN TRUNCATED" << std::endl;
        std::cout << "-- (HLL) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
#if HLL_SEQ
        std::cout << std::endl << "-- STARTING HLL SEQUENTIAL OPERATION --" << std::endl << std::endl;
        y = spmv_HLL_sequential(&hll, x);
        std::cout << std::endl << "-- FINISHED HLL SEQUENTIAL OPERATION --" << std::endl << std::endl;
        if (HLL_SEQ_OUTPUT_LOG)
        {
            std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
            for (IndexType i = 0; i < y.size(); i++)
                std::cout << y[i] << " ";
            std::cout << std::endl;
        }
        y.clear();
        std::cout << std::endl;
#endif
#if HLL
        std::cout << std::endl << "-- STARTING HLL KERNEL OPERATION --" << std::endl << std::endl;
        y = spmv_HLL(&hll, x);
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
        // -------------------------------------------- HYB-ELLG struct
#if HYB_ELL_SEQ || HYB_ELL || HYB_ELLG_SEQ || HYB_ELLG
        std::cout << "-- (HYB-ELL-G) PRE-PROCESSING INPUT --" << std::endl;
        CSR_To_HYBELLG(&csr, &hyb_ellg, HYB_ELLG_LOG);
        std::cout << "-- (HYB-ELL-G) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
#if HYB_ELL_SEQ
        std::cout << std::endl << "-- STARTING HYB_ELL SEQUENTIAL OPERATION --" << std::endl << std::endl;
        y = spmv_HYB_ELL_sequential(&hyb_ellg, x);
        std::cout << std::endl << "-- FINISHED HYB_ELL SEQUENTIAL OPERATION --" << std::endl << std::endl;
        if (HYB_ELL_SEQ_OUTPUT_LOG)
        {
            std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
            for (IndexType i = 0; i < y.size(); i++)
                std::cout << y[i] << " ";
            std::cout << std::endl;
        }
        y.clear();
        std::cout << std::endl;
#endif
#if HYB_ELL
        std::cout << std::endl << "-- STARTING HYB_ELL KERNEL OPERATION --" << std::endl << std::endl;
        y = spmv_HYB_ELL(&hyb_ellg, x);
        std::cout << std::endl << "-- FINISHED HYB_ELL KERNEL OPERATION --" << std::endl << std::endl;
        if (HYB_ELL_OUTPUT_LOG)
        {
            std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
            for (IndexType i = 0; i < y.size(); i++)
                std::cout << y[i] << " ";
            std::cout << std::endl;
        }
        y.clear();
        std::cout << std::endl;
#endif
#if HYB_ELLG_SEQ
        std::cout << std::endl << "-- STARTING HYB_ELL-G SEQUENTIAL OPERATION --" << std::endl << std::endl;
        y = spmv_HYB_ELLG_sequential(&hyb_ellg, x);
        std::cout << std::endl << "-- FINISHED HYB_ELL-G SEQUENTIAL OPERATION --" << std::endl << std::endl;
        if (HYB_ELLG_SEQ_OUTPUT_LOG)
        {
            std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
            for (IndexType i = 0; i < y.size(); i++)
                std::cout << y[i] << " ";
            std::cout << std::endl;
        }
        y.clear();
        std::cout << std::endl;
#endif
#if HYB_ELLG
        std::cout << std::endl << "-- STARTING HYB_ELL-G KERNEL OPERATION --" << std::endl << std::endl;
        y = spmv_HYB_ELLG(&hyb_ellg, x);
        std::cout << std::endl << "-- FINISHED HYB_ELL-G KERNEL OPERATION --" << std::endl << std::endl;
        if (HYB_ELLG_OUTPUT_LOG)
        {
            std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
            for (IndexType i = 0; i < y.size(); i++)
                std::cout << y[i] << " ";
            std::cout << std::endl;
        }
        y.clear();
        std::cout << std::endl;
#endif
        FreeHYBELLG(&hyb_ellg);
#endif
        // -------------------------------------------- HYB-HLL struct
#if HYB_HLL_SEQ || HYB_HLL
        std::cout << "-- (HYB-HLL) PRE-PROCESSING INPUT --" << std::endl;
        CSR_To_HYBHLL(&csr, &hybhll_t, HYB_HLL_LOG);
        std::cout << "-- (HYB-HLL) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
#if HYB_HLL_SEQ
        std::cout << std::endl << "-- STARTING HYB_HLL SEQUENTIAL OPERATION --" << std::endl << std::endl;
        y = spmv_HYB_HLL_sequential(&hybhll_t, x);
        std::cout << std::endl << "-- FINISHED HYB_HLL SEQUENTIAL OPERATION --" << std::endl << std::endl;
        if (HYB_HLL_SEQ_OUTPUT_LOG)
        {
            std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
            for (IndexType i = 0; i < y.size(); i++)
                std::cout << y[i] << " ";
            std::cout << std::endl;
        }
        y.clear();
        std::cout << std::endl;
#endif
#if HYB_HLL
        std::cout << std::endl << "-- STARTING HYB_HLL KERNEL OPERATION --" << std::endl << std::endl;
        y = spmv_HYB_HLL(&hybhll_t, x);
        std::cout << std::endl << "-- FINISHED HYB_HLL KERNEL OPERATION --" << std::endl << std::endl;
        if (HYB_HLL_OUTPUT_LOG)
        {
            std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
            for (IndexType i = 0; i < y.size(); i++)
                std::cout << y[i] << " ";
            std::cout << std::endl;
        }
        y.clear();
        std::cout << std::endl;
#endif
        FreeHYBHLL(&hybhll_t);
#endif
        // -------------------------------------------- JAD struct
#if JAD_SEQ || JAD
        std::cout << "-- (JAD) PRE-PROCESSING INPUT --" << std::endl;
        CSR_To_JAD(&csr, &jad, JAD_LOG);
        std::cout << "-- (JAD) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
#if JAD_SEQ
        std::cout << std::endl << "-- STARTING JAD SEQUENTIAL OPERATION --" << std::endl << std::endl;
        y = spmv_JAD_sequential(&jad, x);
        std::cout << std::endl << "-- FINISHED JAD SEQUENTIAL OPERATION --" << std::endl << std::endl;
        if (JAD_SEQ_OUTPUT_LOG)
        {
            std::cout << std::endl << "-- PRINTING OUTPUT VECTOR RESULTS --" << std::endl;
            for (IndexType i = 0; i < y.size(); i++)
                std::cout << y[i] << " ";
            std::cout << std::endl;
        }
        y.clear();
        std::cout << std::endl;
#endif
#if JAD
        std::cout << std::endl << "-- STARTING JAD KERNEL OPERATION --" << std::endl << std::endl;
        y = spmv_JAD(&jad, x);
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
        x.clear();
    }
#endif
    exit(0);
}
