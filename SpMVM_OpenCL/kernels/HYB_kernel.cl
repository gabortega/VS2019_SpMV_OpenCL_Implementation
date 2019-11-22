#include"../config/compiler_config.h"

#if PRECISION == 2
#define REAL double
#else
#define REAL float
#endif

#include"COO_kernel.cl"
#include"ELL_kernel.cl"

// Implementation based (partly) off the spmv_hyb_device.cu.h from sc2009_spmv by: Nathan Bell & Michael Garland
// URL: https://code.google.com/archive/p/cusp-library/downloads
//
__kernel void spmv_hyb_ell(int n, int nell, int stride,
	__constant int* d_jcoeff,
	__constant REAL* d_a,
	__constant REAL* d_x,
	__global REAL* dst_y)
{
	spmv_ell(n, nell, stride, d_jcoeff, d_a, d_x, dst_y);

}