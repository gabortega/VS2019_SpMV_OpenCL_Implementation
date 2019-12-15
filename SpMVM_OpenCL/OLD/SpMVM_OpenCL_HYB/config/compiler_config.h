#ifndef COMPILER_CONFIG_H
#define COMPILER_CONFIG_H

/*--------------------------------------------------*/
//            Storage format related
//
// This setting sets the precision of the floating point calculations:
// 1 = single-precision
// 2 = double-precision
#define PRECISION 2
// for comparing output
#define ROUNDING_ERROR 1000
//
// This setting is required in order to pre-allocate sufficient space
// for the diag array of the DIA format.
#define MAX_DIAG 10000 // default is 10000
// same but for HDIA
#define MAX_HDIAG 10000 // default is 10000
// same but for ELLG
#define MAX_ELLG 10000 // default is 10000
// same but for HLL
#define MAX_HLL 10000 // default is 10000
//
// HYB format
// taken from sc2009_spmv and altered
// URL: https://code.google.com/archive/p/cusp-library/downloads
//
////////////////////////////////////////////////////////////////////////////////
//! Compute Optimal Number of Columns per Row in the ELL part of the HYB format
//! Examines the distribution of nonzeros per row of the input CSR matrix to find
//! the optimal tradeoff between the ELL and COO portions of the hybrid (HYB)
//! sparse matrix format under the assumption that ELL performance is a fixed
//! multiple of COO performance.  Furthermore, since ELL performance is also
//! sensitive to the absolute number of rows (and COO is not), a threshold is
//! used to ensure that the ELL portion contains enough rows to be worthwhile.
//! The default values were chosen empirically for a GTX280.
//!
//! @param RELATIVE_SPEED       Speed of ELL relative to COO (e.g. 2.0 -> ELL is twice as fast)
//! @param BREAKEVEN_THRESHOLD  Minimum threshold at which ELL is faster than COO
////////////////////////////////////////////////////////////////////////////////
#define RELATIVE_SPEED 3.0 // default is 3.0
#define BREAKEVEN_THRESHOLD 16 // default is 4096
//
// Size of 'hacks' for hacked formats 
// should be multiple of WARP_SIZE
// HLL format
#define HLL_HACKSIZE 32 // default is 32
// HDIA format
#define HDIA_HACKSIZE 32 // default is 32
//
/*--------------------------------------------------*/

/*--------------------------------------------------*/
//            Kernel related
//
#define KERNEL_FOLDER "../kernels"
//
#define COO_KERNEL_FILE "COO_kernel.cl"
#define CSR_KERNEL_FILE "CSR_kernel.cl"
#define JAD_KERNEL_FILE "JAD_kernel.cl"
#define ELL_KERNEL_FILE "ELL_kernel.cl"
#define ELLG_KERNEL_FILE "ELL_kernel.cl"
#define HLL_KERNEL_FILE "ELL_kernel.cl"
#define HLL_LOCAL_KERNEL_FILE "ELL_kernel.cl"
#define DIA_KERNEL_FILE "DIA_kernel.cl"
#define HDIA_KERNEL_FILE "DIA_kernel.cl"
#define HDIA_LOCAL_KERNEL_FILE "DIA_kernel.cl"
#define HYB_ELL_KERNEL_FILE "HYB_kernel.cl"
#define HYB_ELLG_KERNEL_FILE "HYB_kernel.cl"
#define HYB_HLL_KERNEL_FILE "HYB_kernel.cl"
#define HYB_HLL_LOCAL_KERNEL_FILE "HYB_kernel.cl"
//$
#define MAX_THREADS 28*2048 // max consecutive threads for a GTX 1080
#define WARP_SIZE 32
#define WORKGROUP_SIZE 256 // default is 256
#define WARPS_PER_WORKGROUP (WORKGROUP_SIZE / WARP_SIZE)
//
// required for spmv_coo_reduce_update
#define __WORKGROUP_SIZE 512
//
// Repeat kernel operation for evaluating performace
#define REPEAT 100
//
// This setting tunes the memory used by each workgroup in the JAD kernel
// Recommended to use a multiple of the WORKGROUSIZE
#define MAX_NJAD_PER_WG 256 // default is 256
// same but for DIA
#define MAX_NDIAG_PER_WG 256 // default is 256
//
//
// CSR Parameters
#define CSR_WORKGROUP_SIZE 128 // default is 128
//
// Kernels to run (0: Off; 1: On)
#define COO 1
#define CSR 1
#define JAD 1
#define ELL 1
#define ELLG 1
#define HLL 1
#define HLL_LOCAL 1
#define DIA 1
#define HDIA 1
#define HDIA_LOCAL 1
#define HYB_ELL 1
#define HYB_ELLG 1
#define HYB_HLL 1
#define HYB_HLL_LOCAL 1
/*--------------------------------------------------*/

/*--------------------------------------------------*/
//            Input/Output related
//
#define INPUT_FOLDER "../input"
#define INPUT_FILE "dynamicSoaringProblem_1.mtx"
//
#define OUTPUT_FOLDER "../output"
//
// Storage format-specific output folders
//
#define COO_OUTPUT_FOLDER "COO"
#define CSR_OUTPUT_FOLDER "CSR"
#define JAD_OUTPUT_FOLDER "JAD"
#define ELL_OUTPUT_FOLDER "ELL"
#define DIA_OUTPUT_FOLDER "DIA"
#define HYB_OUTPUT_FOLDER "HYB"
//
#define OUTPUT_FILENAME "output"
#define OUTPUT_FILEFORMAT ".txt"
//
// Print out data about each storage format (WARNING: AVOID FOR VERY LARGE MATRICES!)
#define COO_LOG 0
#define CSR_LOG 0
#define JAD_LOG 0
#define ELLG_LOG 0
#define HLL_LOG 0
#define DIA_LOG 0
#define HDIA_LOG 0
#define HYB_ELLG_LOG 0
#define HYB_HLL_LOG 0
//
// Print out output data for each kernel
#define COO_OUTPUT_LOG 1
#define CSR_OUTPUT_LOG 1
#define JAD_OUTPUT_LOG 1
#define ELL_OUTPUT_LOG 1
#define ELLG_OUTPUT_LOG 1
#define HLL_OUTPUT_LOG 1
#define HLL_LOCAL_OUTPUT_LOG 1
#define DIA_OUTPUT_LOG 1
#define HDIA_OUTPUT_LOG 1
#define HDIA_LOCAL_OUTPUT_LOG 1
#define HYB_ELL_OUTPUT_LOG 1
#define HYB_ELLG_OUTPUT_LOG 1
#define HYB_HLL_OUTPUT_LOG 1
#define HYB_HLL_LOCAL_OUTPUT_LOG 1
//
/*--------------------------------------------------*/

#endif