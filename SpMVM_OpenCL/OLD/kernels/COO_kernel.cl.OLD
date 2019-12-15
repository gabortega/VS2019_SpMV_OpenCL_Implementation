#if PRECISION == 2
/*-------------------------------- Double-precision----------------------------------*/
__kernel void spmv_coo_serial(
	__constant unsigned int* d_ir,
	__constant unsigned int* d_jc,
	__constant double* d_val,
	__constant double* d_x,
	__global double* dst_y)
{
#pragma unroll
	for (int n = 0; n < NNZ_MATRIX; n++)
	{
		dst_y[d_ir[n]] += d_val[n] * d_x[d_jc[n]];
	}
}

#else
/*-------------------------------- Single-precision----------------------------------*/
__kernel void spmv_coo_serial(
	__constant unsigned int* d_ir,
	__constant unsigned int* d_jc,
	__constant float* d_val,
	__constant float* d_x,
	__global float* dst_y)
{
#pragma unroll
	for (int n = 0; n < NNZ_MATRIX; n++) 
	{
		dst_y[d_ir[n]] += d_val[n] * d_x[d_jc[n]];
	}
}
#endif