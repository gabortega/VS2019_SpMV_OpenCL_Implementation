#if PRECISION == 2
#define REAL double
#else
#define REAL float
#endif

__kernel void spmv_ell(
#if USE_CONSTANT_MEM
	__constant unsigned int* d_jcoeff,
	__constant REAL* d_a,
	__constant REAL* d_x,
#else
	__global unsigned int* d_jcoeff,
	__global REAL* d_a,
	__global REAL* d_x,
#endif
#if OVERRIDE_MEM
	__global REAL* dst_y,
	__local unsigned int* dummy_mem)
#else
	__global REAL* dst_y)
#endif
{
	__private unsigned int row_id = get_global_id(0);

	__private unsigned int i, j;
	__private REAL r;

	if (row_id >= N_MATRIX) return;

	r = 0.0;
#if NELL >= 32
#pragma unroll(32)
#else
#pragma unroll
#endif
	for (i = 0; i < NELL; i++)
	{
		j = i * STRIDE_MATRIX + row_id;
		r += d_a[j] * d_x[d_jcoeff[j]];
	}
	dst_y[row_id] = r;
}

// OCCUPANCY TEST
__kernel void occ_spmv_ell(
#if USE_CONSTANT_MEM
	__constant unsigned int* d_jcoeff,
	__constant REAL* d_a,
	__constant REAL* d_x,
#else
	__global unsigned int* d_jcoeff,
	__global REAL* d_a,
	__global REAL* d_x,
#endif
#if OVERRIDE_MEM
	__global REAL* dst_y,
	__local unsigned int* dummy_mem)
#else
	__global REAL* dst_y)
#endif
{
	__private unsigned int row_id = get_global_id(0) % N_MATRIX;

	__private unsigned int i, j;
	__private REAL r;

	if (row_id >= N_MATRIX) return;

	r = 0.0;
#if NELL >= 32
#pragma unroll(32)
#else
#pragma unroll
#endif
	for (i = 0; i < NELL; i++)
	{
		j = i * STRIDE_MATRIX + row_id;
		r += d_a[j] * d_x[d_jcoeff[j]];
	}
	if (get_global_id(0) < N_MATRIX)
		dst_y[row_id] = r;
}
