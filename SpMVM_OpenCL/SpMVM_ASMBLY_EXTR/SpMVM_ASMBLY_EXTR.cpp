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

int main(void)
{
    FILE* f;
    struct coo_t coo;
#if GMVM
    struct mat_t mat;
#endif
#if CSR || DIA || TRANSPOSED_DIA || HDIA || ELL || TRANSPOSED_ELL || ELLG || TRANSPOSED_ELLG || HLL || HYB_ELL || HYB_ELLG || HYB_HLL || JAD
    struct csr_t csr;
#if DIA || TRANSPOSED_DIA
    struct dia_t dia;
#endif
#if HDIA || HDIA_OLD
    struct hdia_t hdia;
#endif
#if ELL|| TRANSPOSED_ELL || ELLG|| TRANSPOSED_ELLG
    struct ellg_t ellg;
#endif
#if HLL
    struct hll_t hll;
#endif
#if HYB_ELL || HYB_ELLG
    struct hybellg_t hyb_ellg;
#endif
#if HYB_HLL
    struct hybhll_t hybhll_t;
#endif
#if  JAD
    struct jad_t jad;
#endif
#endif

#if INPUT_FILE_MODE
    std::string input_files = EXTR_RANDOM_INPUT_FILES;
#else
    std::string input_files = EXTR_INPUT_FILES;
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

    std::string output_file;
    
    std::cout << "-- LOADING OPENCL CONFIGURATION " << input_filename << " --" << std::endl;
    cl::Device device = jc::get_device(CL_DEVICE_TYPE_GPU);
    //
    //Print GPU used
    std::string deviceName;
    device.getInfo<std::string>(CL_DEVICE_NAME, &deviceName);
    std::cout << "OpenCL device: " << deviceName << std::endl;
    //
    cl::Context context{ device };
    
    cl::Program program;

    std::string macro;

    std::cout << "-- OPENCL CONFIGURATION LOADED --" << std::endl << std::endl;

    if (createOutputDirectory(OUTPUT_FOLDER, EXTR_OUTPUT_FOLDER))
        exit(1);
    std::cout << "!!! OUTPUT BINARY FILE(S) IS(ARE) BEING WRITTEN TO " << OUTPUT_FOLDER + (std::string)"/" + EXTR_OUTPUT_FOLDER << " !!!" << std::endl;
    std::cout << "!!! PROGRAM WILL EXIT AUTOMATICALLY AFTER PROCESSING; PRESS CTRL-C TO ABORT !!!" << std::endl;
    system("PAUSE");

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

        std::cout << "-- INPUT FILE LOADED --" << std::endl << std::endl;

        // -------------------------------------------- MAT struct
#if  GMVM
        std::cout << "-- (GMVM) PRE-PROCESSING INPUT --" << std::endl;
        COO_To_MAT(&coo, &mat, MAT_LOG);
#if !(CSR || DIA || HDIA || ELL|| TRANSPOSED_ELL || ELLG|| TRANSPOSED_ELLG || HLL || JAD)
        FreeCOO(&coo);
#endif
        std::cout << "-- (GMVM) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
        std::cout << std::endl << "-- STARTING GMVM BINARY EXTRACTION --" << std::endl;
        //
        //Macro
        macro = "-DPRECISION=" + std::to_string(PRECISION) +
            " -DN_MATRIX=" + std::to_string(mat.n) +
            " -DNN_MATRIX=" + std::to_string(mat.n * mat.n) +
            " -DN_WORKGROUPS=" + std::to_string((mat.n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE) +
            " -DWORKGROUP_SIZE=" + std::to_string(WORKGROUP_SIZE);
        //
        program =
            jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + GMVM_KERNEL_FILE, context, device, macro.c_str());
        
        output_file = (OUTPUT_FOLDER + (std::string)"/" + EXTR_OUTPUT_FOLDER + (std::string)"/" + "GMVM_" + input_filenames[run] + getTimeOfRun() + EXTR_OUTPUT_FILEFORMAT);
        dumpPTXCode(program, output_file);
        std::cout << "-- FINISHED GMVM BINARY EXTRACTION --" << std::endl << std::endl;

        FreeMAT(&mat);
#endif
#if CSR || DIA || TRANSPOSED_DIA || HDIA || ELL || TRANSPOSED_ELL || ELLG || TRANSPOSED_ELLG || HLL || JAD
        std::cout << "-- (CSR) PRE-PROCESSING INPUT --" << std::endl;
        COO_To_CSR(&coo, &csr, CSR_LOG);
        FreeCOO(&coo);
        std::cout << "-- (CSR) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
#if CSR
        std::cout << std::endl << "-- STARTING CSR BINARY EXTRACTION --" << std::endl;
        IndexType i, row_len = 0, coop, repeat = 1, nworkgroups;
        for (i = 0; i < csr.n; i++) row_len += csr.ia[i + 1] - csr.ia[i];
        row_len = sqrt(row_len / csr.n);
        for (coop = 1; coop < 32 && row_len >= coop; coop <<= 1);
        nworkgroups = 1 + (csr.n * coop - 1) / (repeat * CSR_WORKGROUP_SIZE);
        if (nworkgroups > 1500)
            for (repeat = 1; (1 + (csr.n * coop - 1) / ((repeat + 1) * CSR_WORKGROUP_SIZE)) > 1500; repeat++);
        nworkgroups = 1 + (csr.n * coop - 1) / (repeat * CSR_WORKGROUP_SIZE);
        //
        //Macro
        macro = "-DPRECISION=" + std::to_string(PRECISION) +
            " -DCSR_REPEAT=" + std::to_string(repeat) +
            " -DCSR_COOP=" + std::to_string(coop) +
            " -DUNROLL_SHARED=" + std::to_string(coop / 4) +
            " -DN_MATRIX=" + std::to_string(csr.n);
        //
        program =
            jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + CSR_KERNEL_FILE, context, device, macro.c_str());

        output_file = (OUTPUT_FOLDER + (std::string)"/" + EXTR_OUTPUT_FOLDER + (std::string)"/" + "CSR_" + input_filenames[run] + getTimeOfRun() + EXTR_OUTPUT_FILEFORMAT);
        dumpPTXCode(program, output_file);
        std::cout << "-- FINISHED CSR BINARY EXTRACTION --" << std::endl << std::endl;
#endif
        // -------------------------------------------- DIA struct
#if DIA || TRANSPOSED_DIA
        std::cout << "-- (DIA) PRE-PROCESSING INPUT --" << std::endl;
        if (!CSR_To_DIA(&csr, &dia, DIA_LOG))
            std::cout << "DIA IS INCOMPLETE" << std::endl;
        std::cout << "-- (DIA) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
#if  DIA
        std::cout << std::endl << "-- STARTING DIA BINARY EXTRACTION --" << std::endl;
        //
        //Macro
        macro = "-DPRECISION=" + std::to_string(PRECISION) +
            " -DN_MATRIX=" + std::to_string(dia.n) +
            " -DSTRIDE_MATRIX=" + std::to_string(dia.stride) +
            " -DWORKGROUP_SIZE=" + std::to_string(WORKGROUP_SIZE) +
            " -DUNROLL_SHARED=" + std::to_string(((WORKGROUP_SIZE + MAX_NDIAG_PER_WG - 1) / MAX_NDIAG_PER_WG) + 1);
        //
        program =
            jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + DIA_KERNEL_FILE, context, device, macro.c_str());

        output_file = (OUTPUT_FOLDER + (std::string)"/" + EXTR_OUTPUT_FOLDER + (std::string)"/" + "DIA_" + input_filenames[run] + getTimeOfRun() + EXTR_OUTPUT_FILEFORMAT);
        dumpPTXCode(program, output_file);
        std::cout << "-- FINISHED DIA BINARY EXTRACTION --" << std::endl << std::endl;
#endif
#if TRANSPOSED_DIA
        std::cout << "-- (TRANSPOSED DIA) PRE-PROCESSING INPUT --" << std::endl;
        transpose_DIA(&dia, TRANSPOSED_DIA_LOG);
        std::cout << "-- (TRANSPOSED DIA) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
        std::cout << std::endl << "-- STARTING TRANSPOSED DIA BINARY EXTRACTION --" << std::endl;
        //
        //Macro
        macro = "-DPRECISION=" + std::to_string(PRECISION) +
            " -DN_MATRIX=" + std::to_string(dia.n) +
            " -DSTRIDE_MATRIX=" + std::to_string(dia.stride) +
            " -DWORKGROUP_SIZE=" + std::to_string(WORKGROUP_SIZE) +
            " -DUNROLL_SHARED=" + std::to_string(((WORKGROUP_SIZE + MAX_NDIAG_PER_WG - 1) / MAX_NDIAG_PER_WG) + 1);
        //
        program =
            jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + TRANSPOSED_DIA_KERNEL_FILE, context, device, macro.c_str());

        output_file = (OUTPUT_FOLDER + (std::string)"/" + EXTR_OUTPUT_FOLDER + (std::string)"/" + "TRANSPOSED_DIA_" + input_filenames[run] + getTimeOfRun() + EXTR_OUTPUT_FILEFORMAT);
        dumpPTXCode(program, output_file);
        std::cout << "-- FINISHED TRANSPOSED DIA BINARY EXTRACTION --" << std::endl << std::endl;
#endif
        FreeDIA(&dia);
#endif
        // -------------------------------------------- HDIA struct
#if HDIA || HDIA_OLD
        std::cout << "-- (HDIA) PRE-PROCESSING INPUT --" << std::endl;
        if (!CSR_To_HDIA(&csr, &hdia, HDIA_LOG))
            std::cout << "HDIA IS INCOMPLETE" << std::endl;
        std::cout << "-- (HDIA) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
#if HDIA
        std::cout << std::endl << "-- STARTING HDIA BINARY EXTRACTION --" << std::endl;
        //
        //Macro
        macro = "-DPRECISION=" + std::to_string(PRECISION) +
            " -DN_MATRIX=" + std::to_string(hdia.n) +
            " -DWORKGROUP_SIZE=" + std::to_string(WORKGROUP_SIZE) +
            " -DMAX_NDIAG=" + std::to_string(MAX_NDIAG_PER_HACK) +
            " -DHACKSIZE=" + std::to_string(HDIA_HACKSIZE) +
            " -DNHOFF=" + std::to_string(hdia.nhoff - 1) +
            " -DUNROLL_SHARED=" + std::to_string(((WORKGROUP_SIZE + MAX_NDIAG_PER_WG - 1) / MAX_NDIAG_PER_WG) + 1);
        //
        program =
            jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + HDIA_KERNEL_FILE, context, device, macro.c_str());

        output_file = (OUTPUT_FOLDER + (std::string)"/" + EXTR_OUTPUT_FOLDER + (std::string)"/" + "HDIA_" + input_filenames[run] + getTimeOfRun() + EXTR_OUTPUT_FILEFORMAT);
        dumpPTXCode(program, output_file);
        std::cout << "-- FINISHED HDIA BINARY EXTRACTION --" << std::endl << std::endl;
#endif
#if HDIA_OLD
        std::cout << std::endl << "-- STARTING HDIA (OLD) BINARY EXTRACTION --" << std::endl;
        //
        IndexType unroll_val;
        for (unroll_val = 1; (*(hdia.ndiags + hdia.nhoff - 1) / 2) >= unroll_val; unroll_val <<= 1);
        //
        //Macro
        macro = "-DPRECISION=" + std::to_string(PRECISION) +
            " -DN_MATRIX=" + std::to_string(hdia.n) +
            " -DHACKSIZE=" + std::to_string(HDIA_HACKSIZE) +
            " -DUNROLL=" + std::to_string(unroll_val);
        //
        program =
            jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + HDIA_OLD_KERNEL_FILE, context, device, macro.c_str());

        output_file = (OUTPUT_FOLDER + (std::string)"/" + EXTR_OUTPUT_FOLDER + (std::string)"/" + "HDIA_OLD_" + input_filenames[run] + getTimeOfRun() + EXTR_OUTPUT_FILEFORMAT);
        dumpPTXCode(program, output_file);
        std::cout << "-- FINISHED HDIA (OLD) BINARY EXTRACTION --" << std::endl << std::endl;
#endif
        FreeHDIA(&hdia);
#endif
        // -------------------------------------------- ELLG struct
#if  ELL || ELLG
        std::cout << "-- (ELL-G) PRE-PROCESSING INPUT --" << std::endl;
        if (!CSR_To_ELLG(&csr, &ellg, ELLG_LOG))
            std::cout << "ELL-G HAS BEEN TRUNCATED" << std::endl;
        std::cout << "-- (ELL-G) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
#if ELL
        std::cout << std::endl << "-- STARTING ELL BINARY EXTRACTION --" << std::endl;
        //
        //Macro
        macro = "-DPRECISION=" + std::to_string(PRECISION) +
            " -DNELL=" + std::to_string(*(ellg.nell + ellg.n)) +
            " -DN_MATRIX=" + std::to_string(ellg.n) +
            " -DSTRIDE_MATRIX=" + std::to_string(ellg.stride);
        //
        program =
            jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + ELL_KERNEL_FILE, context, device, macro.c_str());

        output_file = (OUTPUT_FOLDER + (std::string)"/" + EXTR_OUTPUT_FOLDER + (std::string)"/" + "ELL_" + input_filenames[run] + getTimeOfRun() + EXTR_OUTPUT_FILEFORMAT);
        dumpPTXCode(program, output_file);
        std::cout << "-- FINISHED ELL BINARY EXTRACTION --" << std::endl << std::endl;
#endif
#if ELLG
        std::cout << std::endl << "-- STARTING ELL-G BINARY EXTRACTION --" << std::endl;
        //
        //Macro
        macro = "-DPRECISION=" + std::to_string(PRECISION) +
            " -DN_MATRIX=" + std::to_string(ellg.n) +
            " -DSTRIDE_MATRIX=" + std::to_string(ellg.stride);
        //
        program =
            jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + ELLG_KERNEL_FILE, context, device, macro.c_str());

        output_file = (OUTPUT_FOLDER + (std::string)"/" + EXTR_OUTPUT_FOLDER + (std::string)"/" + "ELLG_" + input_filenames[run] + getTimeOfRun() + EXTR_OUTPUT_FILEFORMAT);
        dumpPTXCode(program, output_file);
        std::cout << "-- FINISHED ELL-G BINARY EXTRACTION --" << std::endl << std::endl;
#endif
#if TRANSPOSED_ELL || TRANSPOSED_ELLG
        transpose_ELLG(&ellg, TRANSPOSED_ELLG_LOG);
#if TRANSPOSED_ELL
        std::cout << std::endl << "-- STARTING TRANSPOSED ELL BINARY EXTRACTION --" << std::endl;
        //
        //Macro
        macro = "-DPRECISION=" + std::to_string(PRECISION) +
            " -DNELL=" + std::to_string(*(ellg.nell + ellg.n)) +
            " -DN_MATRIX=" + std::to_string(ellg.n) +
            " -DSTRIDE_MATRIX=" + std::to_string(ellg.stride);
        //
        program =
            jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + TRANSPOSED_ELL_KERNEL_FILE, context, device, macro.c_str());

        output_file = (OUTPUT_FOLDER + (std::string)"/" + EXTR_OUTPUT_FOLDER + (std::string)"/" + "TRANSPOSED_ELL_" + input_filenames[run] + getTimeOfRun() + EXTR_OUTPUT_FILEFORMAT);
        dumpPTXCode(program, output_file);
        std::cout << "-- FINISHED TRANSPOSED ELL BINARY EXTRACTION --" << std::endl << std::endl;
#endif
#if TRANSPOSED_ELLG
        std::cout << std::endl << "-- STARTING TRANSPOSED ELL-G BINARY EXTRACTION --" << std::endl;
        //
        //Macro
        macro = "-DPRECISION=" + std::to_string(PRECISION) +
            " -DN_MATRIX=" + std::to_string(ellg.n) +
            " -DSTRIDE_MATRIX=" + std::to_string(ellg.stride) +
            " -DMAX_NELL=" + std::to_string(*(ellg.nell + ellg.n));
        //
        program =
            jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + TRANSPOSED_ELLG_KERNEL_FILE, context, device, macro.c_str());

        output_file = (OUTPUT_FOLDER + (std::string)"/" + EXTR_OUTPUT_FOLDER + (std::string)"/" + "TRANSPOSED_ELLG_" + input_filenames[run] + getTimeOfRun() + EXTR_OUTPUT_FILEFORMAT);
        dumpPTXCode(program, output_file);
        std::cout << "-- FINISHED TRANSPOSED ELL-G BINARY EXTRACTION --" << std::endl << std::endl;
#endif
#endif
        FreeELLG(&ellg);
#endif
        // -------------------------------------------- HLL struct
#if  HLL
        std::cout << "-- (HLL) PRE-PROCESSING INPUT --" << std::endl;
        if (!CSR_To_HLL(&csr, &hll, HLL_LOG))
            std::cout << "HLL HAS BEEN TRUNCATED" << std::endl;
        std::cout << "-- (HLL) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
#if HLL
        std::cout << std::endl << "-- STARTING HLL BINARY EXTRACTION --" << std::endl;
        //
        //Macro
        macro = "-DPRECISION=" + std::to_string(PRECISION) +
            " -DHACKSIZE=" + std::to_string(HLL_HACKSIZE) +
            " -DN_MATRIX=" + std::to_string(hll.n) +
            " -DUNROLL=" + std::to_string(unroll_val);
        //
        program =
            jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + HLL_KERNEL_FILE, context, device, macro.c_str());

        output_file = (OUTPUT_FOLDER + (std::string)"/" + EXTR_OUTPUT_FOLDER + (std::string)"/" + "HLL_" + input_filenames[run] + getTimeOfRun() + EXTR_OUTPUT_FILEFORMAT);
        dumpPTXCode(program, output_file);
        std::cout << "-- FINISHED HLL BINARY EXTRACTION --" << std::endl << std::endl;
#endif
        FreeHLL(&hll);
#endif
        // -------------------------------------------- JAD struct
#if JAD
        std::cout << "-- (JAD) PRE-PROCESSING INPUT --" << std::endl;
        CSR_To_JAD(&csr, &jad, JAD_LOG);
        std::cout << "-- (JAD) DONE PRE-PROCESSING INPUT --" << std::endl << std::endl;
        std::cout << std::endl << "-- STARTING JAD BINARY EXTRACTION --" << std::endl;
        //
        //Macro
        macro = "-DPRECISION=" + std::to_string(PRECISION) +
            " -DN_MATRIX=" + std::to_string(jad.n) +
            " -DUNROLL_SHARED=" + std::to_string(((WORKGROUP_SIZE + MAX_NJAD_PER_WG - 1) / MAX_NJAD_PER_WG) + 1) +
            " -DWORKGROUP_SIZE=" + std::to_string(WORKGROUP_SIZE);
        //
        program =
            jc::build_program_from_file(KERNEL_FOLDER + (std::string)"/" + JAD_KERNEL_FILE, context, device, macro.c_str());

        output_file = (OUTPUT_FOLDER + (std::string)"/" + EXTR_OUTPUT_FOLDER + (std::string)"/" + "JAD_" + input_filenames[run] + getTimeOfRun() + EXTR_OUTPUT_FILEFORMAT);
        dumpPTXCode(program, output_file);
        std::cout << "-- FINISHED JAD BINARY EXTRACTION --" << std::endl << std::endl;
        FreeJAD(&jad);
#endif
        FreeCSR(&csr);
#endif
    }
    exit(0);
} 

