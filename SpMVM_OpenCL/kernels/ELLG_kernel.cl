#if PRECISION == 2
#define REAL double
#else
#define REAL float
#endif

__kernel void spmv_ellg(
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
#if OVERRIDE_MEM
	__global REAL* dst_y,
	__local unsigned int* dummy_mem)
#else
	__global REAL* dst_y)
#endif
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
		j = i * STRIDE_MATRIX + row_id;
		r += d_a[j] * d_x[d_jcoeff[j]];
	}
	dst_y[row_id] = r;
}

// OCCUPANCY TEST
__kernel void occ_spmv_ellg(
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
#if OVERRIDE_MEM
	__global REAL* dst_y,
	__local unsigned int* dummy_mem)
#else
	__global REAL* dst_y)
#endif
{
	__private unsigned int row_id = get_global_id(0) % N_MATRIX;

	if (row_id >= N_MATRIX) return;

	__private unsigned int row_nell = d_nell[row_id];

	__private unsigned int i, j;
	__private REAL r;

	r = 0.0;
#pragma unroll(5)
	for (i = 0; i < row_nell; i++)
	{
		j = i * STRIDE_MATRIX + row_id;
		r += d_a[j] * d_x[d_jcoeff[j]];
	}
	if (get_global_id(0) < N_MATRIX)
		dst_y[row_id] = r;
}
