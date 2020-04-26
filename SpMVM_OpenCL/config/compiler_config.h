#ifndef COMPILER_CONFIG_H
#define COMPILER_CONFIG_H

/*--------------------------------------------------*/
//            Storage format related
//
// This setting sets the precision of the floating point calculations:
// 1 = single-precision
// 2 = double-precision
#define PRECISION 1
//
// Use constant memory
#define USE_CONSTANT_MEM 1
//
// Force thread count on all kernels
#define OVERRIDE_THREADS 0
//
// Force local memory use on kernels that don't use local memory
#define OVERRIDE_MEM 0
//
// This setting is required in order to pre-allocate sufficient space
// for the diag array of the DIA format.
#define MAX_DIAG 20480 // default is 20480
// same but for HDIA
#define MAX_HDIAG 20480 // default is 20480
// same but for ELLG
#define MAX_ELLG 20480 // default is 20480
// same but for HLL
#define MAX_HLL 20480 // default is 20480
//
// Size of 'hacks' for hacked formats 
// should be multiple of WARP_SIZE
// HLL format
#define HLL_HACKSIZE 32 // default is 32
// HDIA format
#define HDIA_HACKSIZE 32 // default is 32
//
// ELL-CSR HYB format
// based on: "Accelerating Sparse Matrix Vector Multiplication in Iterative Methods Using GPU"
// by: Kiran Kumar Matam & Kishore Kothapalli
//
#define ELL_ROW_MAX 256 // default is 256; should be a multiple of 32
//
/*--------------------------------------------------*/

/*--------------------------------------------------*/
//            Kernel related
//
#define KERNEL_FOLDER "../kernels"
//
#define COO_KERNEL_FILE "COO_kernel.cl"
#define GMVM_KERNEL_FILE "GMVM_kernel.cl"
#define CSR_KERNEL_FILE "CSR_kernel.cl"
#define JAD_KERNEL_FILE "JAD_kernel.cl"
#define ELL_KERNEL_FILE "ELL_kernel.cl"
#define ELLG_KERNEL_FILE "ELLG_kernel.cl"
#define HLL_KERNEL_FILE "HLL_kernel.cl"
#define DIA_KERNEL_FILE "DIA_kernel.cl"
#define HDIA_KERNEL_FILE "HDIA_kernel.cl"
#define HDIA_OLD_KERNEL_FILE "HDIA_OLD_kernel.cl"
//
#define TRANSPOSED_ELL_KERNEL_FILE "TRANSPOSED_ELL_kernel.cl"
#define TRANSPOSED_ELLG_KERNEL_FILE "TRANSPOSED_ELLG_kernel.cl"
#define TRANSPOSED_DIA_KERNEL_FILE "TRANSPOSED_DIA_kernel.cl"
//
// GPU-specifc parameters
#define CORE_COUNT 20 // GTX 1080: 20 SMs
#define CORE_CLOCK_SPEED 1733000000  // (in Hz) GTX 1080 (Base): 1607MHz || GTX 1080 (Boost): 1733MHz
#define WARP_SIZE 32 // default is 32
#define WORKGROUP_SIZE 256 // default is 256
#define WARPS_PER_WORKGROUP (WORKGROUP_SIZE / WARP_SIZE)
#define MAX_CONC_WG 32 // GTX 1080: maximum 64 conc. workgroups; 32 due to double shared memory
#define MAX_LOCAL_MEM 49152 // GTX 1080: maximum 49152 bytes of local mem. per workgroup
#define MAX_LOCAL_MEM_PER_CORE (MAX_LOCAL_MEM * 2) // GTX 1080: maximum ‭98304‬ bytes of local mem. per SM
#define MAX_WORKGROUP_SIZE 1024 // GTX 1080: maximum 1024 threads per workgroup
#define MAX_OCCUPANCY_TEST_THREAD_COUNT ((MAX_CONC_WG * 2) * MAX_WORKGROUP_SIZE * CORE_COUNT)
#define LOCAL_MEM_CHUNK_SIZE 256
//
// Repeat kernel operation for evaluating performace
#define REPEAT 200
//
// This setting tunes the memory used by each workgroup in the JAD kernel
// Recommended to use a multiple of the WORKGROUSIZE
#define MAX_NJAD_PER_WG 256 // default is 256
// same but for DIA
#define MAX_NDIAG_PER_WG 256 // default is 256
// same but for HDIA
#define MAX_NDIAG_PER_HACK 256 // default is 256
//
// CSR Parameters
#define CSR_WORKGROUP_SIZE 128 // default is 128
#define CSR_WORKGROUP_COUNT_THRESHOLD 1500 // default is 1500
//
// Sequential kernels to run (0: Off; 1: On)
#define GMVM_SEQ 1
//
#define CSR_SEQ 1
//
#define DIA_SEQ 1
#define HDIA_SEQ 1
//
#define ELL_SEQ 1
#define ELLG_SEQ 1
#define HLL_SEQ 1
//
#define HYB_ELL_SEQ 1
#define HYB_ELLG_SEQ 1
#define HYB_HLL_SEQ 1
//
#define JAD_SEQ 1
//
// GPU Kernels to run (0: Off; 1: On)
#define GMVM 1
//
#define CSR 1
//
#define DIA 1
#define HDIA 1
#define HDIA_OLD 1
//
#define ELL 1
#define ELLG 1
#define HLL 1
//
#define HYB_ELL 1
#define HYB_ELLG 1
#define HYB_HLL 1
//
#define JAD 1
//
#define TRANSPOSED_ELL 1
#define TRANSPOSED_ELLG 1
#define TRANSPOSED_DIA 1
//
// Struct-specific categorization
// Sequential kernels
#define SEQ_KERNELS GMVM_SEQ || CSR_SEQ || DIA_SEQ || HDIA_SEQ || ELL_SEQ || ELLG_SEQ || HLL_SEQ || HYB_ELL_SEQ || HYB_ELLG_SEQ || HYB_HLL_SEQ || JAD_SEQ
// Kernel specific structs (without seq. kernels)
#define DIA_STRUCTS_NSEQ DIA || TRANSPOSED_DIA
#define HDIA_STRUCTS_NSEQ HDIA || HDIA_OLD
#define ELLG_STRUCTS_NSEQ ELL || TRANSPOSED_ELL || ELLG || TRANSPOSED_ELLG
#define HLL_STRUCTS_NSEQ HLL
#define HYB_ELLG_STRUCTS_NSEQ HYB_ELL || HYB_ELLG
#define HYB_HLL_STRUCTS_NSEQ HYB_HLL
#define JAD_STRUCTS_NSEQ JAD
// CSR is used as a base for all kernels except GMVM
#define CSR_STRUCTS_NSEQ CSR || DIA_STRUCTS_NSEQ || HDIA_STRUCTS_NSEQ || ELLG_STRUCTS_NSEQ || HLL_STRUCTS_NSEQ || HYB_ELLG_STRUCTS_NSEQ || HYB_HLL_STRUCTS_NSEQ || JAD_STRUCTS_NSEQ
// MAT struct only used in GMVM
#define MAT_STRUCTS_NSEQ GMVM
// COO struct is used for every kernel
#define COO_STRUCTS_NSEQ MAT_STRUCTS_NSEQ || CSR_STRUCTS_NSEQ
//
// Kernel specific structs (with seq. kernels)
#define DIA_STRUCTS DIA_SEQ || DIA_STRUCTS_NSEQ
#define HDIA_STRUCTS HDIA_SEQ || HDIA_STRUCTS_NSEQ
#define ELLG_STRUCTS ELL_SEQ || ELLG_SEQ || ELLG_STRUCTS_NSEQ
#define HLL_STRUCTS HLL_SEQ || HLL_STRUCTS_NSEQ
#define HYB_ELLG_STRUCTS HYB_ELL_SEQ || HYB_ELLG_SEQ || HYB_ELLG_STRUCTS_NSEQ
#define HYB_HLL_STRUCTS HYB_HLL_SEQ || HYB_HLL_STRUCTS_NSEQ
#define JAD_STRUCTS JAD_SEQ || JAD_STRUCTS_NSEQ
// CSR is used as a base for all kernels except GMVM
#define CSR_STRUCTS CSR_SEQ || CSR_STRUCTS_NSEQ
// MAT struct only used in GMVM
#define MAT_STRUCTS GMVM_SEQ || MAT_STRUCTS_NSEQ
// COO struct is used for every kernel
#define COO_STRUCTS MAT_STRUCTS || CSR_STRUCTS
/*--------------------------------------------------*/

/*--------------------------------------------------*/
//            Input/Output related
//
#define INPUT_FILE_MODE 1 // 0 for standard input files (i.e. .../input/); 1 for generated input files (i.e. .../input/random/)
//
#define INPUT_FOLDER "../input"
#define INPUT_FILE "_test_matrix_1.mtx"
#define SUITE_INPUT_FILES "dynamicSoaringProblem_1.mtx;sherman3.mtx;psmigr_1.mtx;msc01050.mtx" //delimiter is ;
#define EXTR_INPUT_FILES "dynamicSoaringProblem_1.mtx;sherman3.mtx;psmigr_1.mtx;msc01050.mtx" //delimiter is ;
#define OCCUPANCY_INPUT_FILES "sherman3.mtx" //delimiter is ;
//
#define GENERATOR_FOLDER "random"
#define RANDOM_INPUT_FILE "imbalanced_cols_zigzag.mtx"
//#define SUITE_RANDOM_INPUT_FILES "random_spread_1.mtx;random_spread_2.mtx;random_spread_3.mtx" //same as above but for randomly generated matrices
//#define SUITE_RANDOM_INPUT_FILES "imbalanced_cols.mtx;imbalanced_cols_inverted.mtx;imbalanced_rows.mtx;imbalanced_rows_inverted.mtx;very_imbalanced_cols.mtx;very_imbalanced_rows.mtx" //same as above but for randomly generated matrices
#define SUITE_RANDOM_INPUT_FILES "random_spread_1.mtx;random_spread_2.mtx;random_spread_3.mtx;random_spread_4.mtx;imbalanced_cols.mtx;imbalanced_cols_zigzag.mtx;imbalanced_cols_inverted.mtx;imbalanced_rows.mtx;imbalanced_rows_zigzag.mtx;imbalanced_rows_inverted.mtx;very_imbalanced_cols.mtx;very_imbalanced_rows.mtx" //same as above but for randomly generated matrices
//
//#define EXTR_RANDOM_INPUT_FILES "random_spread_1.mtx;random_spread_2.mtx;random_spread_3.mtx" //same as above but for randomly generated matrices
//#define EXTR_RANDOM_INPUT_FILES "imbalanced_cols.mtx;imbalanced_cols_inverted.mtx;imbalanced_rows.mtx;imbalanced_rows_inverted.mtx;very_imbalanced_cols.mtx;very_imbalanced_rows.mtx" //same as above but for randomly generated matrices
#define EXTR_RANDOM_INPUT_FILES "random_spread_1.mtx;random_spread_2.mtx;random_spread_3.mtx;random_spread_4.mtx;imbalanced_cols.mtx;imbalanced_cols_zigzag.mtx;imbalanced_cols_inverted.mtx;imbalanced_rows.mtx;imbalanced_rows_zigzag.mtx;imbalanced_rows_inverted.mtx;very_imbalanced_cols.mtx;very_imbalanced_rows.mtx" //same as above but for randomly generated matrices
//
#define OCCUPANCY_RANDOM_INPUT_FILES "random_spread_4.mtx" //same as above but for randomly generated matrices
//
#define OUTPUT_FOLDER "../output"
//
// Storage format-specific output folders
#define COO_OUTPUT_FOLDER "COO"
#define GMVM_OUTPUT_FOLDER "GMVM"
#define CSR_OUTPUT_FOLDER "CSR"
#define JAD_OUTPUT_FOLDER "JAD"
#define ELL_OUTPUT_FOLDER "ELL"
#define DIA_OUTPUT_FOLDER "DIA"
#define HYB_OUTPUT_FOLDER "HYB"
#define SUITE_OUTPUT_FOLDER "SUITE"
#define EXTR_OUTPUT_FOLDER "EXTR"
#define OCCUPANCY_OUTPUT_FOLDER "OCCUPANCY"
//
#define OUTPUT_FILENAME "output"
#define OUTPUT_FILEFORMAT ".txt"
#define EXTR_OUTPUT_FILEFORMAT ".ptx"
//
// Print out data about each storage format (WARNING: AVOID FOR VERY LARGE MATRICES!)
#define COO_LOG 0
#define MAT_LOG 0
#define CSR_LOG 0
#define JAD_LOG 0
#define ELLG_LOG 0
#define HLL_LOG 0
#define DIA_LOG 0
#define HDIA_LOG 0
#define HYB_ELLG_LOG 0
#define HYB_HLL_LOG 0
#define TRANSPOSED_ELLG_LOG 0
#define TRANSPOSED_DIA_LOG 0
//
// Print out output data for each kernel
#define COO_SEQ_OUTPUT_LOG 1
#define COO_OUTPUT_LOG 1
//
#define GMVM_SEQ_OUTPUT_LOG 1
#define GMVM_OUTPUT_LOG 1
//
#define CSR_SEQ_OUTPUT_LOG 1
#define CSR_OUTPUT_LOG 1
//
#define JAD_SEQ_OUTPUT_LOG 1
#define JAD_OUTPUT_LOG 1
//
#define ELL_SEQ_OUTPUT_LOG 1
#define ELL_OUTPUT_LOG 1
#define ELLG_SEQ_OUTPUT_LOG 1
#define ELLG_OUTPUT_LOG 1
#define HLL_SEQ_OUTPUT_LOG 1
#define HLL_OUTPUT_LOG 1
//
#define DIA_SEQ_OUTPUT_LOG 1
#define DIA_OUTPUT_LOG 1
#define HDIA_SEQ_OUTPUT_LOG 1
#define HDIA_OUTPUT_LOG 1
#define HDIA_OLD_OUTPUT_LOG 1
//
#define HYB_ELL_SEQ_OUTPUT_LOG 1
#define HYB_ELL_OUTPUT_LOG 1
#define HYB_ELLG_SEQ_OUTPUT_LOG 1
#define HYB_ELLG_OUTPUT_LOG 1
#define HYB_HLL_SEQ_OUTPUT_LOG 1
#define HYB_HLL_OUTPUT_LOG 1
//
#define TRANSPOSED_ELL_OUTPUT_LOG 1
#define TRANSPOSED_ELLG_OUTPUT_LOG 1
//
#define TRANSPOSED_DIA_OUTPUT_LOG 1
//
/*--------------------------------------------------*/
#endif