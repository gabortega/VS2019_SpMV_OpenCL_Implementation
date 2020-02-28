#if PRECISION == 2
#define REAL double
#else
#define REAL float
#endif

__kernel void spmv_transposed_ellg(
#if USE_CONSTANT_MEM
	__constant unsigned int* d_nell,
	__constant unsigned int* d_jcoeff,
	__constant REAL* d_a,
	__constant REAL* d_x,
#else
	__global unsigned int* d_nell,
	__global unsigned int* d_jcoeff,
	__global REAL* d_a,
	__global REAL* d_x,
#endif
	__global REAL* dst_y)
{
	__private unsigned int row_id = get_global_id(0);

	if (row_id >= N_MATRIX) return;

	__private unsigned int row_nell = d_nell[row_id];

	__private unsigned int i, j;
	__private REAL r;

	r = 0.0;
#pragma unroll(5)
	for (i = 0; i < row_nell; i++)
	{
		j = row_id * MAX_NELL + i;
		r += d_a[j] * d_x[d_jcoeff[j]];
	}
	dst_y[row_id] = r;
}